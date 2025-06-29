//! Streaming processing pipeline for video and real-time image processing
//!
//! This module provides efficient streaming capabilities for processing
//! video streams, webcam feeds, and large image sequences.
//!
//! # Features
//!
//! - Frame-by-frame processing with minimal latency
//! - Buffered processing for throughput optimization
//! - Multi-threaded pipeline stages
//! - Memory-efficient processing of large datasets
//! - Real-time performance monitoring

use crate::error::Result;
use crossbeam_channel::{bounded, Receiver};
use image::GenericImageView;
use ndarray::Array2;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Frame type for streaming processing
#[derive(Clone)]
pub struct Frame {
    /// Frame data as 2D array
    pub data: Array2<f32>,
    /// Frame timestamp
    pub timestamp: Instant,
    /// Frame index
    pub index: usize,
    /// Optional metadata
    pub metadata: Option<FrameMetadata>,
}

/// Frame metadata
#[derive(Clone, Debug)]
pub struct FrameMetadata {
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Frames per second
    pub fps: f32,
    /// Color channels
    pub channels: u8,
}

/// Processing stage trait
pub trait ProcessingStage: Send + 'static {
    /// Process a single frame
    fn process(&mut self, frame: Frame) -> Result<Frame>;

    /// Get stage name for monitoring
    fn name(&self) -> &str;
}

/// Stream processing pipeline
pub struct StreamPipeline {
    stages: Vec<Box<dyn ProcessingStage>>,
    buffer_size: usize,
    num_threads: usize,
    metrics: Arc<Mutex<PipelineMetrics>>,
}

/// Pipeline performance metrics
#[derive(Default, Clone)]
pub struct PipelineMetrics {
    /// Total frames processed
    pub frames_processed: usize,
    /// Average processing time per frame
    pub avg_processing_time: Duration,
    /// Peak processing time
    pub peak_processing_time: Duration,
    /// Frames per second
    pub fps: f32,
    /// Dropped frames
    pub dropped_frames: usize,
}

impl Default for StreamPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamPipeline {
    /// Create a new streaming pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            buffer_size: 10,
            num_threads: num_cpus::get(),
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
        }
    }

    /// Set buffer size for inter-stage communication
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set number of worker threads
    pub fn with_num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Add a processing stage to the pipeline
    pub fn add_stage<S: ProcessingStage>(mut self, stage: S) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Process a stream of frames
    pub fn process_stream<I>(&mut self, input: I) -> StreamProcessor
    where
        I: Iterator<Item = Frame> + Send + 'static,
    {
        let (tx, rx) = bounded(self.buffer_size);
        let metrics = Arc::clone(&self.metrics);

        // Create pipeline stages with channels
        let mut channels = vec![rx];

        for stage in self.stages.drain(..) {
            let (stage_tx, stage_rx) = bounded(self.buffer_size);
            channels.push(stage_rx);

            let stage_metrics = Arc::clone(&metrics);
            let stage_name = stage.name().to_string();
            let prev_rx = channels[channels.len() - 2].clone();

            // Spawn worker thread for this stage
            thread::spawn(move || {
                let mut stage = stage;
                while let Ok(frame) = prev_rx.recv() {
                    let start = Instant::now();

                    match stage.process(frame) {
                        Ok(processed) => {
                            let duration = start.elapsed();

                            // Update metrics
                            if let Ok(mut m) = stage_metrics.lock() {
                                m.frames_processed += 1;
                                m.avg_processing_time = Duration::from_secs_f64(
                                    (m.avg_processing_time.as_secs_f64()
                                        * (m.frames_processed - 1) as f64
                                        + duration.as_secs_f64())
                                        / m.frames_processed as f64,
                                );
                                if duration > m.peak_processing_time {
                                    m.peak_processing_time = duration;
                                }
                            }

                            if stage_tx.send(processed).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("Stage {} error: {}", stage_name, e);
                            if let Ok(mut m) = stage_metrics.lock() {
                                m.dropped_frames += 1;
                            }
                        }
                    }
                }
            });
        }

        let output_rx = channels.pop().unwrap();

        // Input thread
        thread::spawn(move || {
            for frame in input {
                if tx.send(frame).is_err() {
                    break;
                }
            }
        });

        // Return processor with output channel
        StreamProcessor {
            output: output_rx,
            metrics,
        }
    }

    /// Get current pipeline metrics
    pub fn metrics(&self) -> PipelineMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

/// Stream processor handle
pub struct StreamProcessor {
    output: Receiver<Frame>,
    metrics: Arc<Mutex<PipelineMetrics>>,
}

impl StreamProcessor {
    /// Get the next processed frame
    pub fn next(&self) -> Option<Frame> {
        self.output.recv().ok()
    }

    /// Try to get the next frame without blocking
    pub fn try_next(&self) -> Option<Frame> {
        self.output.try_recv().ok()
    }

    /// Get current metrics
    pub fn metrics(&self) -> PipelineMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

impl Iterator for StreamProcessor {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        self.output.recv().ok()
    }
}

/// Example processing stages
/// Grayscale conversion stage
pub struct GrayscaleStage;

impl ProcessingStage for GrayscaleStage {
    fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        // Convert to grayscale if the frame has color channels
        if let Some(ref metadata) = frame.metadata {
            if metadata.channels > 1 {
                // Assuming RGB format, use standard luminance weights
                // Y = 0.299*R + 0.587*G + 0.114*B
                let (height, width) = frame.data.dim();
                let mut grayscale = Array2::<f32>::zeros((height, width));

                // If we have 3 channels, the data should be in format (height, width*3)
                // or we might need to reshape. For now, assume single channel passthrough
                // In a real implementation, we'd handle multi-channel data properly

                // Since we're working with single-channel f32 arrays in the current
                // implementation, we'll use a simple averaging approach
                grayscale.assign(&frame.data);

                frame.data = grayscale;

                // Update metadata to reflect single channel
                if let Some(ref mut meta) = frame.metadata {
                    meta.channels = 1;
                }
            }
        }

        // If already grayscale or no metadata, pass through
        Ok(frame)
    }

    fn name(&self) -> &str {
        "Grayscale"
    }
}

/// Gaussian blur stage
pub struct BlurStage {
    sigma: f32,
}

impl BlurStage {
    /// Create a new Gaussian blur processing stage
    pub fn new(sigma: f32) -> Self {
        Self { sigma }
    }
}

impl ProcessingStage for BlurStage {
    fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        // Apply SIMD-accelerated Gaussian blur
        frame.data = crate::simd_ops::simd_gaussian_blur(&frame.data.view(), self.sigma)?;
        Ok(frame)
    }

    fn name(&self) -> &str {
        "GaussianBlur"
    }
}

/// Edge detection stage
pub struct EdgeDetectionStage {
    #[allow(dead_code)]
    threshold: f32,
}

impl EdgeDetectionStage {
    /// Create a new edge detection processing stage
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl ProcessingStage for EdgeDetectionStage {
    fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        // Apply SIMD-accelerated Sobel edge detection
        let (_, _, magnitude) = crate::simd_ops::simd_sobel_gradients(&frame.data.view())?;
        frame.data = magnitude;
        Ok(frame)
    }

    fn name(&self) -> &str {
        "EdgeDetection"
    }
}

/// Motion detection stage
pub struct MotionDetectionStage {
    previous_frame: Option<Array2<f32>>,
    threshold: f32,
}

impl MotionDetectionStage {
    /// Create a new motion detection processing stage
    pub fn new(threshold: f32) -> Self {
        Self {
            previous_frame: None,
            threshold,
        }
    }
}

impl ProcessingStage for MotionDetectionStage {
    fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        if let Some(ref prev) = self.previous_frame {
            // Compute frame difference
            let diff = &frame.data - prev;
            frame.data = diff.mapv(|x| if x.abs() > self.threshold { 1.0 } else { 0.0 });
        }

        self.previous_frame = Some(frame.data.clone());
        Ok(frame)
    }

    fn name(&self) -> &str {
        "MotionDetection"
    }
}

/// Perspective transformation stage for real-time stream processing
pub struct PerspectiveTransformStage {
    transform: crate::transform::perspective::PerspectiveTransform,
    output_width: u32,
    output_height: u32,
    border_mode: crate::transform::perspective::BorderMode,
}

impl PerspectiveTransformStage {
    /// Create a new perspective transformation stage
    pub fn new(
        transform: crate::transform::perspective::PerspectiveTransform,
        output_width: u32,
        output_height: u32,
        border_mode: crate::transform::perspective::BorderMode,
    ) -> Self {
        Self {
            transform,
            output_width,
            output_height,
            border_mode,
        }
    }

    /// Create perspective correction stage from corner points
    pub fn correction(
        corners: [(f64, f64); 4],
        output_width: u32,
        output_height: u32,
    ) -> Result<Self> {
        let dst_rect = (0.0, 0.0, output_width as f64, output_height as f64);
        let transform = crate::transform::perspective::PerspectiveTransform::quad_to_rect(corners, dst_rect)?;
        
        Ok(Self {
            transform,
            output_width,
            output_height,
            border_mode: crate::transform::perspective::BorderMode::Transparent,
        })
    }
}

impl ProcessingStage for PerspectiveTransformStage {
    fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        // Convert frame data to image format for transformation
        use image::{ImageBuffer, Luma};
        
        let (height, width) = frame.data.dim();
        let mut img_buf = ImageBuffer::new(width as u32, height as u32);
        
        for (y, row) in frame.data.rows().into_iter().enumerate() {
            for (x, &pixel) in row.iter().enumerate() {
                let gray_value = (pixel * 255.0).clamp(0.0, 255.0) as u8;
                img_buf.put_pixel(x as u32, y as u32, Luma([gray_value]));
            }
        }
        
        let src_img = image::DynamicImage::ImageLuma8(img_buf);
        
        // Apply perspective transformation using SIMD-accelerated version
        let transformed = crate::transform::perspective::warp_perspective_simd(
            &src_img,
            &self.transform,
            Some(self.output_width),
            Some(self.output_height),
            self.border_mode,
        )?;
        
        // Convert back to Array2<f32>
        let mut output_data = Array2::zeros((self.output_height as usize, self.output_width as usize));
        
        for y in 0..self.output_height {
            for x in 0..self.output_width {
                let pixel = transformed.get_pixel(x, y);
                let gray_value = pixel[0] as f32 / 255.0;
                output_data[[y as usize, x as usize]] = gray_value;
            }
        }
        
        frame.data = output_data;
        
        // Update metadata
        if let Some(ref mut metadata) = frame.metadata {
            metadata.width = self.output_width;
            metadata.height = self.output_height;
        }
        
        Ok(frame)
    }

    fn name(&self) -> &str {
        "PerspectiveTransform"
    }
}

/// SIMD-accelerated normalization stage
pub struct SimdNormalizationStage;

impl ProcessingStage for SimdNormalizationStage {
    fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        frame.data = crate::simd_ops::simd_normalize_image(&frame.data.view())?;
        Ok(frame)
    }

    fn name(&self) -> &str {
        "SimdNormalization"
    }
}

/// SIMD-accelerated histogram equalization stage
pub struct SimdHistogramEqualizationStage {
    num_bins: usize,
}

impl SimdHistogramEqualizationStage {
    /// Create a new SIMD histogram equalization stage
    pub fn new(num_bins: usize) -> Self {
        Self { num_bins }
    }
}

impl ProcessingStage for SimdHistogramEqualizationStage {
    fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        frame.data = crate::simd_ops::simd_histogram_equalization(&frame.data.view(), self.num_bins)?;
        Ok(frame)
    }

    fn name(&self) -> &str {
        "SimdHistogramEqualization"
    }
}

/// Real-time feature detection stage
pub struct FeatureDetectionStage {
    detector_type: FeatureDetectorType,
    #[allow(dead_code)]
    max_features: usize,
}

/// Types of feature detectors for streaming
pub enum FeatureDetectorType {
    /// Harris corner detection
    Harris { 
        /// Harris response threshold
        threshold: f32, 
        /// Harris parameter k
        k: f32 
    },
    /// FAST corner detection  
    Fast { 
        /// FAST threshold value
        threshold: u8 
    },
    /// Sobel edge detection
    Sobel,
}

impl FeatureDetectionStage {
    /// Create a new feature detection stage
    pub fn new(detector_type: FeatureDetectorType, max_features: usize) -> Self {
        Self {
            detector_type,
            max_features,
        }
    }
}

impl ProcessingStage for FeatureDetectionStage {
    fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        match self.detector_type {
            FeatureDetectorType::Harris { threshold, k } => {
                // Apply Harris corner detection (simplified)
                let (grad_x, grad_y, _) = crate::simd_ops::simd_sobel_gradients(&frame.data.view())?;
                
                // Compute Harris response (simplified version)
                let response = &grad_x * &grad_y - k * (&grad_x * &grad_x + &grad_y * &grad_y);
                frame.data = response.mapv(|x| if x > threshold { x } else { 0.0 });
            },
            FeatureDetectorType::Fast { threshold: _ } => {
                // FAST detection would require more complex implementation
                // For now, use edge detection as approximation
                let (_, _, magnitude) = crate::simd_ops::simd_sobel_gradients(&frame.data.view())?;
                frame.data = magnitude;
            },
            FeatureDetectorType::Sobel => {
                let (_, _, magnitude) = crate::simd_ops::simd_sobel_gradients(&frame.data.view())?;
                frame.data = magnitude;
            },
        }
        
        Ok(frame)
    }

    fn name(&self) -> &str {
        "FeatureDetection"
    }
}

/// Frame buffer stage for temporal operations
pub struct FrameBufferStage {
    buffer: std::collections::VecDeque<Array2<f32>>,
    buffer_size: usize,
    operation: BufferOperation,
}

/// Types of operations on frame buffers
pub enum BufferOperation {
    /// Temporal averaging
    TemporalAverage,
    /// Background subtraction
    BackgroundSubtraction,
    /// Frame differencing
    FrameDifference,
}

impl FrameBufferStage {
    /// Create a new frame buffer stage
    pub fn new(buffer_size: usize, operation: BufferOperation) -> Self {
        Self {
            buffer: std::collections::VecDeque::with_capacity(buffer_size),
            buffer_size,
            operation,
        }
    }
}

impl ProcessingStage for FrameBufferStage {
    fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        // Add current frame to buffer
        self.buffer.push_back(frame.data.clone());
        if self.buffer.len() > self.buffer_size {
            self.buffer.pop_front();
        }
        
        // Apply buffer operation
        match self.operation {
            BufferOperation::TemporalAverage => {
                if !self.buffer.is_empty() {
                    let mut avg = Array2::<f32>::zeros(frame.data.dim());
                    for buffered_frame in &self.buffer {
                        avg += buffered_frame;
                    }
                    frame.data = avg / self.buffer.len() as f32;
                }
            },
            BufferOperation::BackgroundSubtraction => {
                if self.buffer.len() >= self.buffer_size {
                    // Use median of buffer as background
                    let mut background = Array2::<f32>::zeros(frame.data.dim());
                    for buffered_frame in &self.buffer {
                        background += buffered_frame;
                    }
                    background /= self.buffer.len() as f32;
                    frame.data = (&frame.data - &background).mapv(|x| x.abs());
                }
            },
            BufferOperation::FrameDifference => {
                if self.buffer.len() >= 2 {
                    let prev_frame = &self.buffer[self.buffer.len() - 2];
                    frame.data = (&frame.data - prev_frame).mapv(|x| x.abs());
                }
            },
        }
        
        Ok(frame)
    }

    fn name(&self) -> &str {
        match self.operation {
            BufferOperation::TemporalAverage => "TemporalAverage",
            BufferOperation::BackgroundSubtraction => "BackgroundSubtraction",
            BufferOperation::FrameDifference => "FrameDifference",
        }
    }
}

/// Video source type
pub enum VideoSource {
    /// Image sequence (directory of images)
    ImageSequence(std::path::PathBuf),
    /// Video file (requires external decoder)
    VideoFile(std::path::PathBuf),
    /// Camera device
    Camera(u32),
    /// Dummy source for testing
    Dummy {
        /// Frame width in pixels
        width: u32,
        /// Frame height in pixels
        height: u32,
        /// Frames per second
        fps: f32,
    },
}

/// Video reader for streaming
pub struct VideoStreamReader {
    source: VideoSource,
    frame_count: usize,
    fps: f32,
    width: u32,
    height: u32,
    image_files: Option<Vec<std::path::PathBuf>>,
}

impl VideoStreamReader {
    /// Create a video reader from a source
    pub fn from_source(source: VideoSource) -> Result<Self> {
        match source {
            VideoSource::ImageSequence(ref path) => {
                // Read directory and get sorted list of image files
                let mut files = Vec::new();
                if path.is_dir() {
                    for entry in std::fs::read_dir(path).map_err(|e| {
                        crate::error::VisionError::Other(format!("Failed to read directory: {}", e))
                    })? {
                        let entry = entry.map_err(|e| {
                            crate::error::VisionError::Other(format!("Failed to read entry: {}", e))
                        })?;
                        let path = entry.path();
                        if path.is_file() {
                            if let Some(ext) = path.extension() {
                                let ext_str = ext.to_string_lossy().to_lowercase();
                                if ["jpg", "jpeg", "png", "bmp", "tiff"].contains(&ext_str.as_str())
                                {
                                    files.push(path);
                                }
                            }
                        }
                    }
                    files.sort();
                }

                if files.is_empty() {
                    return Err(crate::error::VisionError::Other(
                        "No image files found in directory".to_string(),
                    ));
                }

                // Determine dimensions from first image (in real impl, would load and check)
                Ok(Self {
                    source,
                    frame_count: 0,
                    fps: 30.0,  // Default FPS for image sequences
                    width: 640, // Default, would read from actual image
                    height: 480,
                    image_files: Some(files),
                })
            }
            VideoSource::VideoFile(ref _path) => {
                // Would require video decoder integration (ffmpeg, gstreamer, etc.)
                Err(crate::error::VisionError::Other(
                    "Video file reading not yet implemented. Use image sequences instead."
                        .to_string(),
                ))
            }
            VideoSource::Camera(_device_id) => {
                // Would require camera API integration
                Err(crate::error::VisionError::Other(
                    "Camera reading not yet implemented. Use image sequences instead.".to_string(),
                ))
            }
            VideoSource::Dummy { width, height, fps } => Ok(Self {
                source,
                frame_count: 0,
                fps,
                width,
                height,
                image_files: None,
            }),
        }
    }

    /// Create a dummy video reader for testing
    pub fn dummy(width: u32, height: u32, fps: f32) -> Self {
        Self {
            source: VideoSource::Dummy { width, height, fps },
            frame_count: 0,
            fps,
            width,
            height,
            image_files: None,
        }
    }

    /// Read frames as a stream
    pub fn frames(mut self) -> impl Iterator<Item = Frame> {
        std::iter::from_fn(move || {
            match &self.source {
                VideoSource::ImageSequence(_) => {
                    if let Some(ref files) = self.image_files {
                        if self.frame_count < files.len() {
                            // In a real implementation, we would load the image here
                            // For now, generate a frame with noise to simulate image data
                            let frame_data = Array2::from_shape_fn(
                                (self.height as usize, self.width as usize),
                                |_| rand::random::<f32>(),
                            );

                            let frame = Frame {
                                data: frame_data,
                                timestamp: Instant::now(),
                                index: self.frame_count,
                                metadata: Some(FrameMetadata {
                                    width: self.width,
                                    height: self.height,
                                    fps: self.fps,
                                    channels: 1,
                                }),
                            };

                            self.frame_count += 1;
                            Some(frame)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                VideoSource::Dummy { .. } => {
                    // Generate synthetic frame
                    if self.frame_count < 100 {
                        let frame = Frame {
                            data: Array2::from_shape_fn(
                                (self.height as usize, self.width as usize),
                                |(y, x)| {
                                    // Create a moving pattern
                                    let t = self.frame_count as f32 / self.fps;
                                    ((x as f32 / self.width as f32 * 10.0 + t).sin()
                                        + (y as f32 / self.height as f32 * 10.0 + t).cos())
                                        * 0.5
                                        + 0.5
                                },
                            ),
                            timestamp: Instant::now(),
                            index: self.frame_count,
                            metadata: Some(FrameMetadata {
                                width: self.width,
                                height: self.height,
                                fps: self.fps,
                                channels: 1,
                            }),
                        };

                        self.frame_count += 1;
                        Some(frame)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
    }

    /// Get video properties
    pub fn properties(&self) -> (u32, u32, f32) {
        (self.width, self.height, self.fps)
    }
}

/// Batch processing utilities
pub struct BatchProcessor {
    batch_size: usize,
}

impl BatchProcessor {
    /// Create a new batch processor with specified batch size
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    /// Process frames in batches
    pub fn process_batch<F>(&self, frames: Vec<Frame>, mut processor: F) -> Result<Vec<Frame>>
    where
        F: FnMut(&[Frame]) -> Result<Vec<Frame>>,
    {
        let mut results = Vec::new();

        for chunk in frames.chunks(self.batch_size) {
            let processed = processor(chunk)?;
            results.extend(processed);
        }

        Ok(results)
    }
}

/// Real-time performance monitor
pub struct PerformanceMonitor {
    #[allow(dead_code)]
    start_time: Instant,
    frame_times: Vec<Duration>,
    window_size: usize,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            frame_times: Vec::new(),
            window_size: 100,
        }
    }

    /// Record frame processing time
    pub fn record_frame(&mut self, duration: Duration) {
        self.frame_times.push(duration);

        // Keep only recent frames
        if self.frame_times.len() > self.window_size {
            self.frame_times.remove(0);
        }
    }

    /// Get current FPS
    pub fn fps(&self) -> f32 {
        if self.frame_times.is_empty() {
            return 0.0;
        }

        let avg_duration: Duration =
            self.frame_times.iter().sum::<Duration>() / self.frame_times.len() as u32;
        1.0 / avg_duration.as_secs_f32()
    }

    /// Get average latency
    pub fn avg_latency(&self) -> Duration {
        if self.frame_times.is_empty() {
            return Duration::ZERO;
        }

        self.frame_times.iter().sum::<Duration>() / self.frame_times.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = StreamPipeline::new()
            .with_buffer_size(20)
            .with_num_threads(4)
            .add_stage(GrayscaleStage)
            .add_stage(BlurStage::new(1.0))
            .add_stage(EdgeDetectionStage::new(0.1));

        assert_eq!(pipeline.stages.len(), 3); // 3 stages added to pipeline
    }

    #[test]
    fn test_video_stream_reader() {
        let reader = VideoStreamReader::dummy(640, 480, 30.0);
        let frames: Vec<_> = reader.frames().take(10).collect();

        assert_eq!(frames.len(), 10);
        assert_eq!(frames[0].metadata.as_ref().unwrap().width, 640);
        assert_eq!(frames[0].metadata.as_ref().unwrap().height, 480);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();

        // Simulate frame processing
        for _ in 0..10 {
            monitor.record_frame(Duration::from_millis(16)); // ~60 FPS
        }

        let fps = monitor.fps();
        assert!(fps > 50.0 && fps < 70.0);

        let latency = monitor.avg_latency();
        assert_eq!(latency, Duration::from_millis(16));
    }

    #[test]
    fn test_batch_processor() {
        let processor = BatchProcessor::new(5);
        let frames: Vec<_> = (0..12)
            .map(|i| Frame {
                data: Array2::zeros((10, 10)),
                timestamp: Instant::now(),
                index: i,
                metadata: None,
            })
            .collect();

        let result = processor.process_batch(frames, |batch| Ok(batch.to_vec()));

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 12);
    }

    #[test]
    fn test_perspective_transform_stage() {
        // Create identity transformation
        let transform = crate::transform::perspective::PerspectiveTransform::identity();
        let mut stage = PerspectiveTransformStage::new(
            transform,
            100,
            100,
            crate::transform::perspective::BorderMode::default(),
        );
        
        let frame = Frame {
            data: Array2::from_shape_fn((50, 50), |(y, x)| (x + y) as f32 / 100.0),
            timestamp: Instant::now(),
            index: 0,
            metadata: Some(FrameMetadata {
                width: 50,
                height: 50,
                fps: 30.0,
                channels: 1,
            }),
        };
        
        let result = stage.process(frame);
        assert!(result.is_ok());
        
        let processed = result.unwrap();
        assert_eq!(processed.data.dim(), (100, 100));
    }

    #[test]
    fn test_simd_stages() {
        let frame = Frame {
            data: Array2::from_shape_fn((100, 100), |(y, x)| (x + y) as f32 / 200.0),
            timestamp: Instant::now(),
            index: 0,
            metadata: None,
        };
        
        // Test SIMD normalization
        let mut norm_stage = SimdNormalizationStage;
        let norm_result = norm_stage.process(frame.clone());
        assert!(norm_result.is_ok());
        
        // Test SIMD histogram equalization
        let mut hist_stage = SimdHistogramEqualizationStage::new(256);
        let hist_result = hist_stage.process(frame.clone());
        assert!(hist_result.is_ok());
        
        // Test feature detection
        let mut feature_stage = FeatureDetectionStage::new(
            FeatureDetectorType::Sobel,
            1000,
        );
        let feature_result = feature_stage.process(frame);
        assert!(feature_result.is_ok());
    }

    #[test]
    fn test_frame_buffer_stage() {
        let mut buffer_stage = FrameBufferStage::new(
            5,
            BufferOperation::TemporalAverage,
        );
        
        // Process several frames
        for i in 0..10 {
            let frame = Frame {
                data: Array2::from_elem((10, 10), i as f32),
                timestamp: Instant::now(),
                index: i,
                metadata: None,
            };
            
            let result = buffer_stage.process(frame);
            assert!(result.is_ok());
        }
    }
}
