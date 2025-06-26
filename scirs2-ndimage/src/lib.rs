//! N-dimensional image processing module
//!
//! This module provides functions for processing and analyzing n-dimensional arrays as images.
//! It includes filters, interpolation, measurements, morphology, feature detection, and segmentation functions.

// Public modules
pub mod backend;
pub mod chunked;
pub mod error;
pub mod features;
pub mod filters;
pub mod interpolation;
pub mod measurements;
pub mod memory_management;
pub mod morphology;
pub mod segmentation;
pub mod streaming;
pub mod threading;

// Re-exports
pub use self::error::*;

// Feature detection module exports
pub use self::features::{
    canny, edge_detector, edge_detector_simple, fast_corners, gradient_edges, harris_corners,
    laplacian_edges, sobel_edges, EdgeDetectionAlgorithm, EdgeDetectionConfig, GradientMethod,
    // Machine learning-based detection
    LearnedEdgeDetector, LearnedKeypointDescriptor, SemanticFeatureExtractor,
    ObjectProposalGenerator, ObjectProposal, MLDetectorConfig,
    FeatureDetectorWeights, BatchNormParams,
};

// Filters module exports
pub use self::filters::{
    bilateral_filter, convolve, filter_functions, gaussian_filter, gaussian_filter_chunked,
    gaussian_filter_f32, gaussian_filter_f64, generic_filter, laplace, maximum_filter,
    median_filter, median_filter_chunked, minimum_filter, percentile_filter, rank_filter, sobel,
    uniform_filter, uniform_filter_chunked, BorderMode,
};

#[cfg(feature = "simd")]
pub use self::filters::{bilateral_filter_simd_f32, bilateral_filter_simd_f64};

// Segmentation module exports
pub use self::segmentation::{
    adaptive_threshold, marker_watershed, otsu_threshold, threshold_binary, watershed,
    AdaptiveMethod,
    // Advanced segmentation algorithms
    graph_cuts, GraphCutsParams, InteractiveGraphCuts,
    active_contour, create_circle_contour, create_ellipse_contour, 
    mask_to_contour, smooth_contour, ActiveContourParams,
    chan_vese, chan_vese_multiphase, mask_to_level_set, 
    checkerboard_level_set, ChanVeseParams,
};

// Interpolation module exports
pub use self::interpolation::{
    affine_transform, bspline, geometric_transform, map_coordinates, rotate, shift, spline_filter,
    spline_filter1d, value_at_coordinates, zoom, BoundaryMode, InterpolationOrder,
};

// Measurements module exports
pub use self::measurements::{
    center_of_mass, count_labels, extrema, find_objects, local_extrema, mean_labels, moments,
    moments_inertia_tensor, peak_prominences, peak_widths, region_properties, sum_labels,
    variance_labels, RegionProperties,
};

// Morphology module exports
pub use self::morphology::{
    binary_closing, binary_dilation, binary_erosion, binary_fill_holes, binary_hit_or_miss,
    binary_opening, black_tophat, box_structure, disk_structure, find_boundaries,
    generate_binary_structure, grey_closing, grey_dilation, grey_erosion, grey_opening,
    iterate_structure, label, morphological_gradient, morphological_laplace, remove_small_holes,
    remove_small_objects, white_tophat, Connectivity, MorphBorderMode,
};

// Memory management exports
pub use self::memory_management::{
    check_memory_limit, create_output_array, estimate_memory_usage, BufferPool, InPlaceOp,
    MemoryConfig, MemoryEfficientOp, MemoryStrategy,
};

// Chunked processing exports
pub use self::chunked::{process_chunked, ChunkConfig, ChunkProcessor, GaussianChunkProcessor};

// Backend exports
pub use self::backend::{
    auto_backend, Backend, BackendBuilder, BackendConfig, BackendExecutor, BackendOp,
};

// Threading exports
pub use self::threading::{
    configure_parallel_ops, get_thread_pool_config, init_thread_pool, update_thread_pool_config,
    AdaptiveThreadPool, ThreadPoolConfig, ThreadPoolContext, WorkStealingQueue, WorkerInfo,
};

// Streaming exports
pub use self::streaming::{
    stream_process_file, StreamConfig, StreamProcessor, StreamableOp, StreamingGaussianFilter,
};
