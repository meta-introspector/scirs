//! Visualization tools for neural networks
//!
//! This module provides comprehensive visualization capabilities including:
//! - Network architecture visualization with interactive graphs
//! - Training curves and metrics plotting
//! - Layer activation maps and feature visualization
//! - Attention mechanisms visualization
//! - Interactive dashboards and real-time monitoring

use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
// Model trait import temporarily commented out
use ndarray::{Array2, ArrayD};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::path::PathBuf;

/// Visualization configuration
#[derive(Debug, Clone, Serialize)]
pub struct VisualizationConfig {
    /// Output directory for generated visualizations
    pub output_dir: PathBuf,
    /// Image format for static outputs
    pub image_format: ImageFormat,
    /// Interactive visualization settings
    pub interactive: InteractiveConfig,
    /// Color scheme and styling
    pub style: StyleConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
}

/// Supported image formats for visualization output
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum ImageFormat {
    /// PNG format (recommended for web)
    PNG,
    /// SVG format (vector graphics)
    SVG,
    /// PDF format (publication quality)
    PDF,
    /// HTML format (interactive)
    HTML,
    /// JSON format (data export)
    JSON,
}

/// Interactive visualization configuration
#[derive(Debug, Clone, Serialize)]
pub struct InteractiveConfig {
    /// Enable interactive features
    pub enable_interaction: bool,
    /// Web server port for live visualization
    pub server_port: u16,
    /// Auto-refresh interval in milliseconds
    pub refresh_interval_ms: u32,
    /// Enable real-time updates
    pub real_time_updates: bool,
    /// Maximum data points to display
    pub max_data_points: usize,
}

/// Style configuration for visualizations
#[derive(Debug, Clone, Serialize)]
pub struct StyleConfig {
    /// Color palette
    pub color_palette: ColorPalette,
    /// Font settings
    pub font: FontConfig,
    /// Layout settings
    pub layout: LayoutConfig,
    /// Theme selection
    pub theme: Theme,
}

/// Color palette for visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ColorPalette {
    /// Default SciRS2 colors
    Default,
    /// Colorblind-friendly palette
    ColorblindFriendly,
    /// High contrast palette
    HighContrast,
    /// Grayscale palette
    Grayscale,
    /// Custom palette
    Custom(Vec<String>),
}

/// Font configuration
#[derive(Debug, Clone, Serialize)]
pub struct FontConfig {
    /// Font family
    pub family: String,
    /// Base font size
    pub size: u32,
    /// Title font size multiplier
    pub title_scale: f32,
    /// Label font size multiplier
    pub label_scale: f32,
}

/// Layout configuration
#[derive(Debug, Clone, Serialize)]
pub struct LayoutConfig {
    /// Canvas width
    pub width: u32,
    /// Canvas height
    pub height: u32,
    /// Margin settings
    pub margins: Margins,
    /// Grid settings
    pub grid: GridConfig,
}

/// Margin configuration
#[derive(Debug, Clone, Serialize)]
pub struct Margins {
    /// Top margin
    pub top: u32,
    /// Bottom margin
    pub bottom: u32,
    /// Left margin
    pub left: u32,
    /// Right margin
    pub right: u32,
}

/// Grid configuration
#[derive(Debug, Clone, Serialize)]
pub struct GridConfig {
    /// Show grid lines
    pub show_grid: bool,
    /// Grid line color
    pub grid_color: String,
    /// Grid line width
    pub grid_width: u32,
    /// Grid opacity
    pub grid_opacity: f32,
}

/// Visualization theme
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Theme {
    /// Light theme
    Light,
    /// Dark theme
    Dark,
    /// Auto theme (system preference)
    Auto,
    /// Custom theme
    Custom(CustomTheme),
}

/// Custom theme configuration
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CustomTheme {
    /// Background color
    pub background: String,
    /// Text color
    pub text: String,
    /// Primary accent color
    pub primary: String,
    /// Secondary accent color
    pub secondary: String,
    /// Success color
    pub success: String,
    /// Warning color
    pub warning: String,
    /// Error color
    pub error: String,
}

/// Performance configuration for visualizations
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceConfig {
    /// Maximum number of points per plot
    pub max_points_per_plot: usize,
    /// Enable data downsampling
    pub enable_downsampling: bool,
    /// Downsampling strategy
    pub downsampling_strategy: DownsamplingStrategy,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size limit in MB
    pub cache_size_mb: usize,
}

/// Downsampling strategy for large datasets
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum DownsamplingStrategy {
    /// Take every nth point
    Uniform,
    /// Largest triangle three bucket algorithm
    LTTB,
    /// Min-max decimation
    MinMax,
    /// Statistical sampling
    Statistical,
}

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

/// Training metrics visualizer
#[allow(dead_code)]
pub struct TrainingVisualizer<F: Float + Debug> {
    /// Training history
    metrics_history: Vec<TrainingMetrics<F>>,
    /// Visualization configuration
    config: VisualizationConfig,
    /// Active plots
    active_plots: HashMap<String, PlotConfig>,
}

/// Training metrics for a single epoch/step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics<F: Float + Debug> {
    /// Epoch number
    pub epoch: usize,
    /// Step number within epoch
    pub step: usize,
    /// Timestamp
    pub timestamp: String,
    /// Loss values
    pub losses: HashMap<String, F>,
    /// Accuracy metrics
    pub accuracies: HashMap<String, F>,
    /// Learning rate
    pub learning_rate: F,
    /// Other custom metrics
    pub custom_metrics: HashMap<String, F>,
    /// System metrics
    pub system_metrics: SystemMetrics,
}

/// System performance metrics during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// GPU memory usage in MB (if available)
    pub gpu_memory_mb: Option<f64>,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// GPU utilization percentage (if available)
    pub gpu_utilization: Option<f64>,
    /// Training step duration in milliseconds
    pub step_duration_ms: f64,
    /// Samples processed per second
    pub samples_per_second: f64,
}

/// Plot configuration
#[derive(Debug, Clone, Serialize)]
pub struct PlotConfig {
    /// Plot title
    pub title: String,
    /// X-axis configuration
    pub x_axis: AxisConfig,
    /// Y-axis configuration
    pub y_axis: AxisConfig,
    /// Series to plot
    pub series: Vec<SeriesConfig>,
    /// Plot type
    pub plot_type: PlotType,
    /// Update mode
    pub update_mode: UpdateMode,
}

/// Axis configuration
#[derive(Debug, Clone, Serialize)]
pub struct AxisConfig {
    /// Axis label
    pub label: String,
    /// Axis scale
    pub scale: AxisScale,
    /// Range (None for auto)
    pub range: Option<(f64, f64)>,
    /// Show grid lines
    pub show_grid: bool,
    /// Tick configuration
    pub ticks: TickConfig,
}

/// Axis scale type
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AxisScale {
    /// Linear scale
    Linear,
    /// Logarithmic scale
    Log,
    /// Square root scale
    Sqrt,
    /// Custom scale
    Custom(String),
}

/// Tick configuration
#[derive(Debug, Clone, Serialize)]
pub struct TickConfig {
    /// Tick interval (None for auto)
    pub interval: Option<f64>,
    /// Tick format
    pub format: TickFormat,
    /// Show tick labels
    pub show_labels: bool,
    /// Tick rotation angle
    pub rotation: f32,
}

/// Tick format options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum TickFormat {
    /// Automatic formatting
    Auto,
    /// Fixed decimal places
    Fixed(u32),
    /// Scientific notation
    Scientific,
    /// Percentage
    Percentage,
    /// Custom format string
    Custom(String),
}

/// Data series configuration
#[derive(Debug, Clone, Serialize)]
pub struct SeriesConfig {
    /// Series name
    pub name: String,
    /// Data source (metric name)
    pub data_source: String,
    /// Line style
    pub style: LineStyleConfig,
    /// Marker style
    pub markers: MarkerConfig,
    /// Series color
    pub color: String,
    /// Series opacity
    pub opacity: f32,
}

/// Line style configuration for series
#[derive(Debug, Clone, Serialize)]
pub struct LineStyleConfig {
    /// Line style
    pub style: LineStyle,
    /// Line width
    pub width: f32,
    /// Smoothing enabled
    pub smoothing: bool,
    /// Smoothing window size
    pub smoothing_window: usize,
}

/// Marker configuration for data points
#[derive(Debug, Clone, Serialize)]
pub struct MarkerConfig {
    /// Show markers
    pub show: bool,
    /// Marker shape
    pub shape: MarkerShape,
    /// Marker size
    pub size: f32,
    /// Marker fill color
    pub fill_color: String,
    /// Marker border color
    pub border_color: String,
}

/// Marker shape options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum MarkerShape {
    /// Circle marker
    Circle,
    /// Square marker
    Square,
    /// Triangle marker
    Triangle,
    /// Diamond marker
    Diamond,
    /// Cross marker
    Cross,
    /// Plus marker
    Plus,
}

/// Plot type options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum PlotType {
    /// Line plot
    Line,
    /// Scatter plot
    Scatter,
    /// Bar plot
    Bar,
    /// Area plot
    Area,
    /// Histogram
    Histogram,
    /// Box plot
    Box,
    /// Heatmap
    Heatmap,
}

/// Update mode for plots
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum UpdateMode {
    /// Append new data
    Append,
    /// Replace all data
    Replace,
    /// Rolling window
    Rolling(usize),
}

/// Layer activation visualizer
#[allow(dead_code)]
pub struct ActivationVisualizer<F: Float + Debug + ndarray::ScalarOperand> {
    /// Model reference
    model: Sequential<F>,
    /// Visualization configuration
    config: VisualizationConfig,
    /// Cached activations
    activation_cache: HashMap<String, ArrayD<F>>,
}

/// Activation visualization options
#[derive(Debug, Clone, Serialize)]
pub struct ActivationVisualizationOptions {
    /// Layers to visualize
    pub target_layers: Vec<String>,
    /// Visualization type
    pub visualization_type: ActivationVisualizationType,
    /// Normalization method
    pub normalization: ActivationNormalization,
    /// Color mapping
    pub colormap: Colormap,
    /// Aggregation method for multi-channel data
    pub aggregation: ChannelAggregation,
}

/// Types of activation visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ActivationVisualizationType {
    /// Feature maps as heatmaps
    FeatureMaps,
    /// Activation histograms
    Histograms,
    /// Statistics summary
    Statistics,
    /// Spatial attention maps
    AttentionMaps,
    /// Activation flow
    ActivationFlow,
}

/// Activation normalization methods
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ActivationNormalization {
    /// No normalization
    None,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Percentile-based normalization
    Percentile(f64, f64),
    /// Custom normalization function
    Custom(String),
}

/// Color mapping for visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Colormap {
    /// Viridis colormap
    Viridis,
    /// Plasma colormap
    Plasma,
    /// Inferno colormap
    Inferno,
    /// Jet colormap
    Jet,
    /// Grayscale
    Gray,
    /// Red-blue diverging
    RdBu,
    /// Custom colormap
    Custom(Vec<String>),
}

/// Channel aggregation methods
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ChannelAggregation {
    /// No aggregation (show all channels)
    None,
    /// Average across channels
    Mean,
    /// Maximum across channels
    Max,
    /// Minimum across channels
    Min,
    /// Standard deviation across channels
    Std,
    /// Select specific channels
    Select(Vec<usize>),
}

/// Attention mechanism visualizer
#[allow(dead_code)]
pub struct AttentionVisualizer<F: Float + Debug + ndarray::ScalarOperand> {
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
        // TODO: Implement model structure analysis
        // This would inspect the Sequential model and extract layer information
        Err(NeuralError::NotImplementedError(
            "Model structure analysis not yet implemented".to_string(),
        ))
    }

    fn select_layout_algorithm(&self, _layer_info: &[LayerInfo]) -> LayoutAlgorithm {
        // For now, default to hierarchical layout
        // In a full implementation, this would analyze the network structure
        // and choose the most appropriate layout algorithm
        LayoutAlgorithm::Hierarchical
    }

    fn compute_hierarchical_layout(
        &self,
        _layer_info: &[LayerInfo],
    ) -> Result<(Vec<LayerPosition>, Vec<Connection>)> {
        // TODO: Implement hierarchical layout algorithm
        Err(NeuralError::NotImplementedError(
            "Hierarchical layout not yet implemented".to_string(),
        ))
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

    fn create_svg_content(&self, _layout: &NetworkLayout) -> Result<String> {
        // TODO: Implement SVG generation
        let svg_template = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
  <title>Neural Network Architecture</title>
  <defs>
    <style>
      .layer {{ fill: #4CAF50; stroke: #2E7D32; stroke-width: 2; }}
      .connection {{ stroke: #666; stroke-width: 1.5; fill: none; }}
      .layer-text {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
    </style>
  </defs>
  
  <!-- Network visualization content would be generated here -->
  <text x="50%" y="50%" class="layer-text">Network visualization not yet implemented</text>
</svg>"#,
            self.config.style.layout.width, self.config.style.layout.height
        );

        Ok(svg_template)
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
}

impl<F: Float + Debug + 'static + num_traits::FromPrimitive + Send + Sync> TrainingVisualizer<F> {
    /// Create a new training visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            metrics_history: Vec::new(),
            config,
            active_plots: HashMap::new(),
        }
    }

    /// Add training metrics for visualization
    pub fn add_metrics(&mut self, metrics: TrainingMetrics<F>) {
        self.metrics_history.push(metrics);

        // Apply downsampling if needed
        if self.metrics_history.len() > self.config.performance.max_points_per_plot
            && self.config.performance.enable_downsampling
        {
            self.downsample_metrics();
        }
    }

    /// Generate training curves visualization
    pub fn visualize_training_curves(&self) -> Result<Vec<PathBuf>> {
        let mut output_files = Vec::new();

        // Generate loss curves
        if let Some(loss_plot) = self.create_loss_plot()? {
            let loss_path = self.config.output_dir.join("training_loss.html");
            fs::write(&loss_path, loss_plot)
                .map_err(|e| NeuralError::IOError(format!("Failed to write loss plot: {}", e)))?;
            output_files.push(loss_path);
        }

        // Generate accuracy curves
        if let Some(accuracy_plot) = self.create_accuracy_plot()? {
            let accuracy_path = self.config.output_dir.join("training_accuracy.html");
            fs::write(&accuracy_path, accuracy_plot).map_err(|e| {
                NeuralError::IOError(format!("Failed to write accuracy plot: {}", e))
            })?;
            output_files.push(accuracy_path);
        }

        // Generate learning rate plot
        if let Some(lr_plot) = self.create_learning_rate_plot()? {
            let lr_path = self.config.output_dir.join("learning_rate.html");
            fs::write(&lr_path, lr_plot).map_err(|e| {
                NeuralError::IOError(format!("Failed to write learning rate plot: {}", e))
            })?;
            output_files.push(lr_path);
        }

        // Generate system metrics plot
        if let Some(system_plot) = self.create_system_metrics_plot()? {
            let system_path = self.config.output_dir.join("system_metrics.html");
            fs::write(&system_path, system_plot).map_err(|e| {
                NeuralError::IOError(format!("Failed to write system metrics plot: {}", e))
            })?;
            output_files.push(system_path);
        }

        Ok(output_files)
    }

    fn downsample_metrics(&mut self) {
        // TODO: Implement downsampling based on strategy
        match self.config.performance.downsampling_strategy {
            DownsamplingStrategy::Uniform => {
                // Keep every nth point
                let step = self.metrics_history.len() / self.config.performance.max_points_per_plot;
                if step > 1 {
                    let mut downsampled = Vec::new();
                    for (i, metric) in self.metrics_history.iter().enumerate() {
                        if i % step == 0 {
                            downsampled.push(metric.clone());
                        }
                    }
                    self.metrics_history = downsampled;
                }
            }
            _ => {
                // For now, just truncate to max size
                if self.metrics_history.len() > self.config.performance.max_points_per_plot {
                    let start =
                        self.metrics_history.len() - self.config.performance.max_points_per_plot;
                    self.metrics_history.drain(0..start);
                }
            }
        }
    }

    fn create_loss_plot(&self) -> Result<Option<String>> {
        if self.metrics_history.is_empty() {
            return Ok(None);
        }

        // TODO: Implement actual plotting library integration
        // For now, return a placeholder HTML
        let plot_html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>Training Loss</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="lossPlot" style="width:100%;height:500px;"></div>
    <script>
        // TODO: Implement actual loss curve plotting
        var trace = {
            x: [1, 2, 3, 4],
            y: [0.8, 0.6, 0.4, 0.3],
            type: 'scatter',
            name: 'Training Loss'
        };
        
        var layout = {
            title: 'Training Loss Over Time',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Loss' }
        };
        
        Plotly.newPlot('lossPlot', [trace], layout);
    </script>
</body>
</html>"#;

        Ok(Some(plot_html.to_string()))
    }

    fn create_accuracy_plot(&self) -> Result<Option<String>> {
        if self.metrics_history.is_empty() {
            return Ok(None);
        }

        // TODO: Implement accuracy plotting
        Ok(None)
    }

    fn create_learning_rate_plot(&self) -> Result<Option<String>> {
        if self.metrics_history.is_empty() {
            return Ok(None);
        }

        // TODO: Implement learning rate plotting
        Ok(None)
    }

    fn create_system_metrics_plot(&self) -> Result<Option<String>> {
        if self.metrics_history.is_empty() {
            return Ok(None);
        }

        // TODO: Implement system metrics plotting
        Ok(None)
    }
}

impl<
        F: Float + Debug + 'static + num_traits::FromPrimitive + ndarray::ScalarOperand + Send + Sync,
    > ActivationVisualizer<F>
{
    /// Create a new activation visualizer
    pub fn new(model: Sequential<F>, config: VisualizationConfig) -> Self {
        Self {
            model,
            config,
            activation_cache: HashMap::new(),
        }
    }

    /// Visualize layer activations for given input
    pub fn visualize_activations(
        &mut self,
        input: &ArrayD<F>,
        options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // Compute activations
        self.compute_activations(input, &options.target_layers)?;

        // Generate visualizations based on type
        match options.visualization_type {
            ActivationVisualizationType::FeatureMaps => self.generate_feature_maps(options),
            ActivationVisualizationType::Histograms => self.generate_histograms(options),
            ActivationVisualizationType::Statistics => self.generate_statistics(options),
            ActivationVisualizationType::AttentionMaps => self.generate_attention_maps(options),
            ActivationVisualizationType::ActivationFlow => self.generate_activation_flow(options),
        }
    }

    fn compute_activations(&mut self, _input: &ArrayD<F>, _target_layers: &[String]) -> Result<()> {
        // TODO: Implement activation computation
        // This would run forward pass and capture intermediate activations
        Err(NeuralError::NotImplementedError(
            "Activation computation not yet implemented".to_string(),
        ))
    }

    fn generate_feature_maps(
        &self,
        _options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement feature map generation
        Err(NeuralError::NotImplementedError(
            "Feature map visualization not yet implemented".to_string(),
        ))
    }

    fn generate_histograms(
        &self,
        _options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement histogram generation
        Err(NeuralError::NotImplementedError(
            "Histogram visualization not yet implemented".to_string(),
        ))
    }

    fn generate_statistics(
        &self,
        _options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement statistics generation
        Err(NeuralError::NotImplementedError(
            "Statistics visualization not yet implemented".to_string(),
        ))
    }

    fn generate_attention_maps(
        &self,
        _options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement attention map generation
        Err(NeuralError::NotImplementedError(
            "Attention map visualization not yet implemented".to_string(),
        ))
    }

    fn generate_activation_flow(
        &self,
        _options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement activation flow generation
        Err(NeuralError::NotImplementedError(
            "Activation flow visualization not yet implemented".to_string(),
        ))
    }
}

impl<
        F: Float + Debug + 'static + num_traits::FromPrimitive + ndarray::ScalarOperand + Send + Sync,
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

    fn extract_attention_patterns(&mut self, _input: &ArrayD<F>) -> Result<()> {
        // TODO: Implement attention extraction
        // This would run forward pass and extract attention weights from attention layers
        Err(NeuralError::NotImplementedError(
            "Attention extraction not yet implemented".to_string(),
        ))
    }

    fn generate_attention_heatmap(
        &self,
        _options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement attention heatmap generation
        Err(NeuralError::NotImplementedError(
            "Attention heatmap not yet implemented".to_string(),
        ))
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
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            output_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            image_format: ImageFormat::SVG,
            interactive: InteractiveConfig::default(),
            style: StyleConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        Self {
            enable_interaction: true,
            server_port: 8080,
            refresh_interval_ms: 1000,
            real_time_updates: true,
            max_data_points: 10000,
        }
    }
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            color_palette: ColorPalette::Default,
            font: FontConfig::default(),
            layout: LayoutConfig::default(),
            theme: Theme::Light,
        }
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: "Arial, sans-serif".to_string(),
            size: 12,
            title_scale: 1.5,
            label_scale: 0.9,
        }
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            margins: Margins::default(),
            grid: GridConfig::default(),
        }
    }
}

impl Default for Margins {
    fn default() -> Self {
        Self {
            top: 40,
            bottom: 60,
            left: 80,
            right: 40,
        }
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            show_grid: true,
            grid_color: "#e0e0e0".to_string(),
            grid_width: 1,
            grid_opacity: 0.5,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_points_per_plot: 10000,
            enable_downsampling: true,
            downsampling_strategy: DownsamplingStrategy::Uniform,
            enable_caching: true,
            cache_size_mb: 100,
        }
    }
}

// Serialization support is now provided by derive macros

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use crate::models::sequential::Sequential;
    use rand::SeedableRng;
    use tempfile::TempDir;

    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        assert_eq!(config.image_format, ImageFormat::SVG);
        assert!(config.interactive.enable_interaction);
        assert_eq!(config.style.color_palette, ColorPalette::Default);
        assert_eq!(config.performance.max_points_per_plot, 10000);
    }

    #[test]
    fn test_network_visualizer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let mut model: Sequential<f32> = Sequential::new();
        let dense = Dense::new(10, 1, Some("relu"), &mut rng).unwrap();
        model.add_layer(dense);

        let mut config = VisualizationConfig::default();
        config.output_dir = temp_dir.path().to_path_buf();

        let visualizer = NetworkVisualizer::new(model, config);
        assert!(visualizer.layout_cache.is_none());
    }

    #[test]
    fn test_training_visualizer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = VisualizationConfig::default();
        config.output_dir = temp_dir.path().to_path_buf();

        let visualizer: TrainingVisualizer<f64> = TrainingVisualizer::new(config);
        assert_eq!(visualizer.metrics_history.len(), 0);
        assert_eq!(visualizer.active_plots.len(), 0);
    }

    #[test]
    fn test_training_metrics_creation() {
        let mut losses = HashMap::new();
        losses.insert("cross_entropy".to_string(), 0.5);

        let mut accuracies = HashMap::new();
        accuracies.insert("top1".to_string(), 0.85);

        let system_metrics = SystemMetrics {
            memory_usage_mb: 512.0,
            gpu_memory_mb: Some(1024.0),
            cpu_utilization: 75.0,
            gpu_utilization: Some(90.0),
            step_duration_ms: 150.0,
            samples_per_second: 64.0,
        };

        let metrics = TrainingMetrics {
            epoch: 10,
            step: 1000,
            timestamp: "2023-01-01T00:00:00Z".to_string(),
            losses,
            accuracies,
            learning_rate: 0.001,
            custom_metrics: HashMap::new(),
            system_metrics,
        };

        assert_eq!(metrics.epoch, 10);
        assert_eq!(metrics.step, 1000);
        assert_eq!(metrics.losses["cross_entropy"], 0.5);
        assert_eq!(metrics.accuracies["top1"], 0.85);
    }

    #[test]
    fn test_activation_visualizer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let mut model: Sequential<f32> = Sequential::new();
        let dense = Dense::new(10, 1, Some("relu"), &mut rng).unwrap();
        model.add_layer(dense);

        let mut config = VisualizationConfig::default();
        config.output_dir = temp_dir.path().to_path_buf();

        let visualizer = ActivationVisualizer::new(model, config);
        assert_eq!(visualizer.activation_cache.len(), 0);
    }

    #[test]
    fn test_attention_visualizer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let mut model: Sequential<f32> = Sequential::new();
        let dense = Dense::new(10, 1, Some("relu"), &mut rng).unwrap();
        model.add_layer(dense);

        let mut config = VisualizationConfig::default();
        config.output_dir = temp_dir.path().to_path_buf();

        let visualizer = AttentionVisualizer::new(model, config);
        assert_eq!(visualizer.attention_cache.len(), 0);
    }

    #[test]
    fn test_color_palette_types() {
        let default_palette = ColorPalette::Default;
        let colorblind_palette = ColorPalette::ColorblindFriendly;
        let custom_palette =
            ColorPalette::Custom(vec!["#FF0000".to_string(), "#00FF00".to_string()]);

        assert_eq!(default_palette, ColorPalette::Default);
        assert_eq!(colorblind_palette, ColorPalette::ColorblindFriendly);
        if let ColorPalette::Custom(colors) = custom_palette {
            assert_eq!(colors.len(), 2);
            assert_eq!(colors[0], "#FF0000");
        }
    }

    #[test]
    fn test_export_options() {
        let export_options = ExportOptions {
            format: ExportFormat::Image(ImageFormat::PNG),
            quality: ExportQuality::High,
            resolution: Resolution {
                width: 1920,
                height: 1080,
                dpi: 300,
            },
            include_metadata: true,
            compression: CompressionSettings {
                enabled: true,
                level: 6,
                lossless: false,
            },
        };

        assert_eq!(export_options.quality, ExportQuality::High);
        assert_eq!(export_options.resolution.width, 1920);
        assert!(export_options.include_metadata);
        assert!(export_options.compression.enabled);
    }

    #[test]
    fn test_downsampling_strategies() {
        let strategies = vec![
            DownsamplingStrategy::Uniform,
            DownsamplingStrategy::LTTB,
            DownsamplingStrategy::MinMax,
            DownsamplingStrategy::Statistical,
        ];

        assert_eq!(strategies.len(), 4);
        assert_eq!(strategies[0], DownsamplingStrategy::Uniform);
        assert_eq!(strategies[1], DownsamplingStrategy::LTTB);
    }

    #[test]
    fn test_bounding_box_computation() {
        let positions = vec![
            LayerPosition {
                name: "layer1".to_string(),
                layer_type: "Dense".to_string(),
                position: Point2D { x: 0.0, y: 0.0 },
                size: Size2D {
                    width: 10.0,
                    height: 5.0,
                },
                io_info: LayerIOInfo {
                    input_shape: vec![10],
                    output_shape: vec![5],
                    parameter_count: 55,
                    flops: 110,
                },
                visual_props: LayerVisualProps {
                    fill_color: "#4CAF50".to_string(),
                    border_color: "#2E7D32".to_string(),
                    border_width: 2.0,
                    opacity: 1.0,
                    icon: None,
                },
            },
            LayerPosition {
                name: "layer2".to_string(),
                layer_type: "Dense".to_string(),
                position: Point2D { x: 20.0, y: 10.0 },
                size: Size2D {
                    width: 8.0,
                    height: 4.0,
                },
                io_info: LayerIOInfo {
                    input_shape: vec![5],
                    output_shape: vec![1],
                    parameter_count: 6,
                    flops: 12,
                },
                visual_props: LayerVisualProps {
                    fill_color: "#2196F3".to_string(),
                    border_color: "#1976D2".to_string(),
                    border_width: 2.0,
                    opacity: 1.0,
                    icon: None,
                },
            },
        ];

        let mut config = VisualizationConfig::default();
        config.output_dir = PathBuf::from("/tmp");

        let model: Sequential<f32> = Sequential::new();
        let visualizer = NetworkVisualizer::new(model, config);
        let bounds = visualizer.compute_bounds(&positions);

        assert_eq!(bounds.min_x, -5.0); // 0.0 - 10.0/2
        assert_eq!(bounds.max_x, 24.0); // 20.0 + 8.0/2
        assert_eq!(bounds.min_y, -2.5); // 0.0 - 5.0/2
        assert_eq!(bounds.max_y, 12.0); // 10.0 + 4.0/2
    }
}
