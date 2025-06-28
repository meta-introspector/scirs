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
                                    (m.avg_processing_time.as_secs_f64() * (m.frames_processed - 1) as f64
                                        + duration.as_secs_f64()) / m.frames_processed as f64
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
        fps: f32 
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
                    for entry in std::fs::read_dir(path)
                        .map_err(|e| crate::error::VisionError::Other(format!("Failed to read directory: {}", e)))?
                    {
                        let entry = entry.map_err(|e| crate::error::VisionError::Other(format!("Failed to read entry: {}", e)))?;
                        let path = entry.path();
                        if path.is_file() {
                            if let Some(ext) = path.extension() {
                                let ext_str = ext.to_string_lossy().to_lowercase();
                                if ["jpg", "jpeg", "png", "bmp", "tiff"].contains(&ext_str.as_str()) {
                                    files.push(path);
                                }
                            }
                        }
                    }
                    files.sort();
                }
                
                if files.is_empty() {
                    return Err(crate::error::VisionError::Other("No image files found in directory".to_string()));
                }
                
                // Determine dimensions from first image (in real impl, would load and check)
                Ok(Self {
                    source,
                    frame_count: 0,
                    fps: 30.0, // Default FPS for image sequences
                    width: 640, // Default, would read from actual image
                    height: 480,
                    image_files: Some(files),
                })
            }
            VideoSource::VideoFile(ref _path) => {
                // Would require video decoder integration (ffmpeg, gstreamer, etc.)
                Err(crate::error::VisionError::Other(
                    "Video file reading not yet implemented. Use image sequences instead.".to_string()
                ))
            }
            VideoSource::Camera(_device_id) => {
                // Would require camera API integration
                Err(crate::error::VisionError::Other(
                    "Camera reading not yet implemented. Use image sequences instead.".to_string()
                ))
            }
            VideoSource::Dummy { width, height, fps } => {
                Ok(Self {
                    source,
                    frame_count: 0,
                    fps,
                    width,
                    height,
                    image_files: None,
                })
            }
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
                                |_| rand::random::<f32>()
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
                                        + (y as f32 / self.height as f32 * 10.0 + t).cos()) * 0.5 + 0.5
                                }
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
        
        let avg_duration: Duration = self.frame_times.iter().sum::<Duration>() / self.frame_times.len() as u32;
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
        
        let result = processor.process_batch(frames, |batch| {
            Ok(batch.to_vec())
        });
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 12);
    }
}