//! N-dimensional image processing module
//!
//! This module provides functions for processing and analyzing n-dimensional arrays as images.
//! It includes filters, interpolation, measurements, morphology, feature detection, and segmentation functions.

// Public modules
pub mod analysis;
pub mod backend;
pub mod chunked;
pub mod chunked_v2;
pub mod domain_specific;
pub mod error;
pub mod features;
pub mod filters;
pub mod interpolation;
pub mod measurements;
pub mod memory_management;
pub mod mmap_io;
pub mod morphology;
pub mod profiling;
pub mod scipy_compat;
pub mod segmentation;
pub mod streaming;
pub mod threading;
pub mod visualization;

// Re-exports
pub use self::error::*;

// Feature detection module exports
pub use self::features::{
    canny,
    edge_detector,
    edge_detector_simple,
    fast_corners,
    gradient_edges,
    harris_corners,
    laplacian_edges,
    sobel_edges,
    BatchNormParams,
    EdgeDetectionAlgorithm,
    EdgeDetectionConfig,
    FeatureDetectorWeights,
    GradientMethod,
    // Machine learning-based detection
    LearnedEdgeDetector,
    LearnedKeypointDescriptor,
    MLDetectorConfig,
    ObjectProposal,
    ObjectProposalGenerator,
    SemanticFeatureExtractor,
};

// Filters module exports
pub use self::filters::{
    // Advanced filters
    adaptive_wiener_filter,
    anisotropic_diffusion,
    bilateral_filter,
    bilateral_gradient_filter,
    coherence_enhancing_diffusion,
    convolve,
    // Wavelets
    dwt_1d,
    dwt_2d,
    filter_functions,
    gabor_filter,
    gabor_filter_bank,
    gaussian_filter,
    gaussian_filter_chunked,
    gaussian_filter_f32,
    gaussian_filter_f64,
    generic_filter,
    idwt_1d,
    idwt_2d,
    laplace,
    log_gabor_filter,
    maximum_filter,
    median_filter,
    median_filter_chunked,
    minimum_filter,
    non_local_means,
    percentile_filter,
    rank_filter,
    shock_filter,
    sobel,
    steerable_filter,
    uniform_filter,
    uniform_filter_chunked,
    wavelet_decompose,
    wavelet_denoise,
    wavelet_reconstruct,
    BorderMode,
    GaborParams,
    WaveletFamily,
    WaveletFilter,
};

#[cfg(feature = "simd")]
pub use self::filters::{bilateral_filter_simd_f32, bilateral_filter_simd_f64};

// Segmentation module exports
pub use self::segmentation::{
    active_contour,
    adaptive_threshold,
    chan_vese,
    chan_vese_multiphase,
    checkerboard_level_set,
    create_circle_contour,
    create_ellipse_contour,
    // Advanced segmentation algorithms
    graph_cuts,
    marker_watershed,
    mask_to_contour,
    mask_to_level_set,
    otsu_threshold,
    smooth_contour,
    threshold_binary,
    watershed,
    ActiveContourParams,
    AdaptiveMethod,
    ChanVeseParams,
    GraphCutsParams,
    InteractiveGraphCuts,
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

// Domain-specific imaging exports
pub use self::domain_specific::{
    medical::{
        detect_lung_nodules, enhance_bone_structure, frangi_vesselness, Nodule,
        VesselEnhancementParams,
    },
    microscopy::{
        colocalization_analysis, detect_nuclei, segment_cells, CellInfo, CellSegmentationParams,
        ColocalizationMetrics, ThresholdMethod,
    },
    satellite::{compute_ndvi, detect_clouds, detect_water_bodies, pan_sharpen, PanSharpenMethod},
};

// Analysis module exports
pub use self::analysis::{
    batch_quality_assessment, compute_local_variance, contrast_to_noise_ratio,
    estimate_fractal_dimension, image_entropy, image_quality_assessment, image_sharpness,
    local_feature_analysis, mean_absolute_error, mean_squared_error, multi_scale_analysis,
    peak_signal_to_noise_ratio, signal_to_noise_ratio, structural_similarity_index,
    texture_analysis, ImageQualityMetrics, MultiScaleConfig, TextureMetrics,
};

// SIMD-optimized analysis functions
#[cfg(feature = "simd")]
pub use self::analysis::{compute_moments_simd_f32, image_quality_assessment_simd_f32};

// Parallel analysis functions
#[cfg(feature = "parallel")]
pub use self::analysis::image_entropy_parallel;

// Visualization module exports
pub use self::visualization::{
    create_colormap, create_image_montage, generate_report, plot_contour, plot_heatmap,
    plot_histogram, plot_profile, plot_statistical_comparison, plot_surface, visualize_gradient,
    ColorMap, PlotConfig, ReportConfig, ReportFormat,
};
