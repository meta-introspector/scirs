//! Image segmentation module
//!
//! This module provides functions for segmenting images into regions
//! or partitioning images into meaningful parts.

mod thresholding;
mod watershed;
mod graph_cuts;
mod active_contours;
mod chan_vese;

// Re-export submodule components
pub use self::thresholding::{
    adaptive_threshold, otsu_threshold, threshold_binary, AdaptiveMethod,
};
pub use self::watershed::{marker_watershed, watershed};

// Advanced segmentation algorithms
pub use self::graph_cuts::{
    graph_cuts, GraphCutsParams, InteractiveGraphCuts,
};
pub use self::active_contours::{
    active_contour, create_circle_contour, create_ellipse_contour, 
    mask_to_contour, smooth_contour, ActiveContourParams,
};
pub use self::chan_vese::{
    chan_vese, chan_vese_multiphase, mask_to_level_set, 
    checkerboard_level_set, ChanVeseParams,
};
