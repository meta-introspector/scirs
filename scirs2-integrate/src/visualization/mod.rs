//! Visualization utilities for numerical integration and specialized solvers
//!
//! This module provides tools for visualizing results from various solvers,
//! including phase space plots, bifurcation diagrams, and field visualizations.

pub mod types;
pub mod engine;
pub mod utils;
pub mod interactive;
pub mod advanced;
pub mod specialized;
pub mod error_viz;

// Re-export all public types for backward compatibility
pub use types::{
    PhaseSpacePlot, BifurcationDiagram, PlotMetadata, VectorFieldPlot,
    HeatMapPlot, SurfacePlot, OutputFormat, ColorScheme, PlotStatistics,
    ParameterExplorationPlot, AttractorStability, RealTimeBifurcationPlot,
    PhaseSpace3D, SensitivityPlot, InteractivePlotControls,
    ExplorationMethod, ParameterRegion, ParameterExplorationResult,
    ExplorationMetrics, AttractorInfo, DimensionReductionMethod,
    ClusteringMethod, HighDimensionalPlot, AnimationSettings,
    FluidState, FluidState3D, ErrorVisualizationOptions, ErrorType,
    ErrorDistributionPlot, ErrorStatistics, ConvergencePlot,
    MultiMetricConvergencePlot, ConvergenceCurve, StepSizeAnalysisPlot,
    PhaseDensityPlot,
};

// Re-export from engine module
pub use engine::VisualizationEngine;

// Re-export from utils module
pub use utils::{
    generate_colormap, optimal_grid_resolution, plot_statistics,
};

// Re-export from interactive module
pub use interactive::{
    InteractiveParameterExplorer, BifurcationDiagramGenerator,
};

// Re-export from advanced module
pub use advanced::{
    MultiDimensionalVisualizer, AnimatedVisualizer,
    advanced_visualization, advanced_interactive_3d,
};

// Re-export from specialized module
pub use specialized::{
    QuantumVisualizer, FluidVisualizer, FinanceVisualizer,
    specialized_visualizations,
};

// Re-export from error_viz module
pub use error_viz::{
    ErrorVisualizationEngine, ConvergenceVisualizer,
    ConvergenceVisualizationEngine, PerformanceTracker,
    MetricStatistics, ConvergenceInfo,
};