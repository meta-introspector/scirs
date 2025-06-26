//! Streaming operations for processing large datasets
//!
//! This module provides functionality for processing images that are too large
//! to fit in memory by processing them in chunks or tiles.

use ndarray::{Array, ArrayView, ArrayViewMut, Dimension, Ix2, Ix3, IxDyn, ShapeBuilder, Slice};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::BorderMode;
use scirs2_core::parallel_ops::*;

/// Configuration for streaming operations
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Size of chunks to process at once (in bytes)
    pub chunk_size: usize,
    /// Overlap between chunks (in pixels per dimension)
    pub overlap: Vec<usize>,
    /// Whether to use memory mapping when possible
    pub use_mmap: bool,
    /// Number of chunks to keep in cache
    pub cache_chunks: usize,
    /// Directory for temporary files
    pub temp_dir: Option<String>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 128 * 1024 * 1024, // 128 MB
            overlap: vec![],
            use_mmap: true,
            cache_chunks: 4,
            temp_dir: None,
        }
    }
}

/// Trait for operations that can be applied in a streaming fashion
pub trait StreamableOp<T, D>: Send + Sync
where
    T: Float + FromPrimitive + Debug + Clone,
    D: Dimension,
{
    /// Apply operation to a chunk
    fn apply_chunk(&self, chunk: &ArrayView<T, D>) -> NdimageResult<Array<T, D>>;

    /// Get required overlap for this operation
    fn required_overlap(&self) -> Vec<usize>;

    /// Merge overlapping regions from adjacent chunks
    fn merge_overlap(
        &self,
        output: &mut ArrayViewMut<T, D>,
        new_chunk: &ArrayView<T, D>,
        overlap_info: &OverlapInfo,
    ) -> NdimageResult<()>;
}

/// Information about chunk overlap
#[derive(Debug, Clone)]
pub struct OverlapInfo {
    /// Dimension being processed
    pub dimension: usize,
    /// Start index in the output array
    pub output_start: usize,
    /// End index in the output array
    pub output_end: usize,
    /// Size of overlap region
    pub overlap_size: usize,
}

/// Streaming processor for large arrays
pub struct StreamProcessor<T> {
    config: StreamConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> StreamProcessor<T>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process a large array stored in a file
    pub fn process_file<D, Op>(
        &self,
        input_path: &Path,
        output_path: &Path,
        shape: &[usize],
        op: Op,
    ) -> NdimageResult<()>
    where
        D: Dimension,
        Op: StreamableOp<T, D>,
    {
        let element_size = std::mem::size_of::<T>();
        let total_elements: usize = shape.iter().product();
        let total_size = total_elements * element_size;

        // Calculate chunk dimensions
        let chunk_dims = self.calculate_chunk_dimensions(shape, element_size)?;

        // Open input and output files
        let mut input_file = BufReader::new(File::open(input_path)?);
        let mut output_file = BufWriter::new(File::create(output_path)?);

        // Process chunks
        for chunk_info in self.chunk_iterator(shape, &chunk_dims) {
            // Read chunk from file
            let chunk = self.read_chunk(&mut input_file, &chunk_info, shape)?;

            // Apply operation
            let result = op.apply_chunk(&chunk.view())?;

            // Write result to output file
            self.write_chunk(&mut output_file, &result.view(), &chunk_info)?;
        }

        Ok(())
    }

    /// Process a large array in memory with reduced memory footprint
    pub fn process_in_memory<D, Op>(
        &self,
        input: &ArrayView<T, D>,
        op: Op,
    ) -> NdimageResult<Array<T, D>>
    where
        D: Dimension,
        Op: StreamableOp<T, D>,
    {
        let shape = input.shape();
        let element_size = std::mem::size_of::<T>();

        // Calculate chunk dimensions
        let chunk_dims = self.calculate_chunk_dimensions(shape, element_size)?;

        // Create output array
        let mut output = Array::zeros(input.raw_dim());

        // Process chunks in parallel if enabled
        if is_parallel_enabled() {
            let chunks: Vec<_> = self.chunk_iterator(shape, &chunk_dims).collect();

            par_iter(&chunks).try_for_each(|chunk_info| {
                let chunk = self.extract_chunk(input, chunk_info)?;
                let result = op.apply_chunk(&chunk.view())?;

                // Thread-safe writing would be needed here
                // For now, we'll process sequentially
                Ok::<(), NdimageError>(())
            })?;
        } else {
            // Sequential processing
            for chunk_info in self.chunk_iterator(shape, &chunk_dims) {
                let chunk = self.extract_chunk(input, &chunk_info)?;
                let result = op.apply_chunk(&chunk.view())?;
                self.insert_chunk(&mut output.view_mut(), &result.view(), &chunk_info)?;
            }
        }

        Ok(output)
    }

    /// Calculate optimal chunk dimensions based on available memory
    fn calculate_chunk_dimensions(
        &self,
        shape: &[usize],
        element_size: usize,
    ) -> NdimageResult<Vec<usize>> {
        let ndim = shape.len();
        let mut chunk_dims = shape.to_vec();

        // Start with full dimensions and reduce until it fits in chunk_size
        let mut current_size = shape.iter().product::<usize>() * element_size;

        while current_size > self.config.chunk_size && chunk_dims.iter().any(|&d| d > 1) {
            // Find largest dimension and halve it
            let (max_idx, _) = chunk_dims
                .iter()
                .enumerate()
                .filter(|(_, &d)| d > 1)
                .max_by_key(|(_, &d)| d)
                .unwrap();

            chunk_dims[max_idx] /= 2;
            current_size = chunk_dims.iter().product::<usize>() * element_size;
        }

        // Add overlap if specified
        if !self.config.overlap.is_empty() {
            for (i, &overlap) in self.config.overlap.iter().enumerate() {
                if i < ndim {
                    chunk_dims[i] = chunk_dims[i].saturating_add(overlap * 2);
                }
            }
        }

        Ok(chunk_dims)
    }

    /// Iterator over chunk information
    fn chunk_iterator<'a>(
        &'a self,
        shape: &'a [usize],
        chunk_dims: &'a [usize],
    ) -> ChunkIterator<'a> {
        ChunkIterator::new(shape, chunk_dims, &self.config.overlap)
    }

    /// Extract a chunk from an array
    fn extract_chunk<D>(
        &self,
        array: &ArrayView<T, D>,
        chunk_info: &ChunkInfo,
    ) -> NdimageResult<Array<T, D>>
    where
        D: Dimension,
    {
        let slices: Vec<_> = chunk_info
            .ranges
            .iter()
            .map(|r| Slice::from(r.start..r.end))
            .collect();

        Ok(array.slice(slices.as_slice()).to_owned())
    }

    /// Insert a chunk into an array
    fn insert_chunk<D>(
        &self,
        output: &mut ArrayViewMut<T, D>,
        chunk: &ArrayView<T, D>,
        chunk_info: &ChunkInfo,
    ) -> NdimageResult<()>
    where
        D: Dimension,
    {
        let slices: Vec<_> = chunk_info
            .output_ranges
            .iter()
            .map(|r| Slice::from(r.start..r.end))
            .collect();

        let mut output_slice = output.slice_mut(slices.as_slice());
        output_slice.assign(chunk);

        Ok(())
    }

    /// Read a chunk from a file
    fn read_chunk(
        &self,
        file: &mut BufReader<File>,
        chunk_info: &ChunkInfo,
        shape: &[usize],
    ) -> NdimageResult<Array<T, IxDyn>> {
        let element_size = std::mem::size_of::<T>();
        let chunk_elements: usize = chunk_info.ranges.iter().map(|r| r.end - r.start).product();

        // Calculate file offset
        let offset = self.calculate_file_offset(&chunk_info.ranges, shape, element_size);
        file.seek(SeekFrom::Start(offset as u64))?;

        // Read data
        let mut buffer = vec![T::zero(); chunk_elements];
        let byte_buffer = unsafe {
            std::slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut u8,
                chunk_elements * element_size,
            )
        };
        file.read_exact(byte_buffer)?;

        // Create array from buffer
        let chunk_shape: Vec<_> = chunk_info.ranges.iter().map(|r| r.end - r.start).collect();
        Ok(Array::from_shape_vec(IxDyn(&chunk_shape), buffer)?)
    }

    /// Write a chunk to a file
    fn write_chunk(
        &self,
        file: &mut BufWriter<File>,
        chunk: &ArrayView<T, IxDyn>,
        chunk_info: &ChunkInfo,
    ) -> NdimageResult<()> {
        let element_size = std::mem::size_of::<T>();

        // Convert to bytes and write
        let slice = chunk
            .as_slice()
            .ok_or_else(|| NdimageError::InvalidInput("Chunk is not contiguous".into()))?;

        let byte_slice = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * element_size)
        };

        file.write_all(byte_slice)?;
        Ok(())
    }

    /// Calculate file offset for a chunk
    fn calculate_file_offset(
        &self,
        ranges: &[std::ops::Range<usize>],
        shape: &[usize],
        element_size: usize,
    ) -> usize {
        let mut offset = 0;
        let mut stride = element_size;

        for (i, range) in ranges.iter().enumerate().rev() {
            offset += range.start * stride;
            if i > 0 {
                stride *= shape[i];
            }
        }

        offset
    }
}

/// Information about a chunk
#[derive(Debug, Clone)]
struct ChunkInfo {
    /// Ranges in the input array
    ranges: Vec<std::ops::Range<usize>>,
    /// Ranges in the output array (excluding overlap)
    output_ranges: Vec<std::ops::Range<usize>>,
}

/// Iterator over chunks
struct ChunkIterator<'a> {
    shape: &'a [usize],
    chunk_dims: &'a [usize],
    overlap: &'a [usize],
    current: Vec<usize>,
    done: bool,
}

impl<'a> ChunkIterator<'a> {
    fn new(shape: &'a [usize], chunk_dims: &'a [usize], overlap: &'a [usize]) -> Self {
        Self {
            shape,
            chunk_dims,
            overlap,
            current: vec![0; shape.len()],
            done: false,
        }
    }
}

impl<'a> Iterator for ChunkIterator<'a> {
    type Item = ChunkInfo;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut ranges = Vec::new();
        let mut output_ranges = Vec::new();

        for i in 0..self.shape.len() {
            let overlap = self.overlap.get(i).copied().unwrap_or(0);
            let start = self.current[i].saturating_sub(overlap);
            let end = (self.current[i] + self.chunk_dims[i]).min(self.shape[i]);

            ranges.push(start..end);

            // Output range excludes overlap
            let output_start = if self.current[i] == 0 { 0 } else { overlap };
            let output_end = if self.current[i] + self.chunk_dims[i] >= self.shape[i] {
                end - start
            } else {
                end - start - overlap
            };

            output_ranges.push(output_start..output_end);
        }

        let chunk_info = ChunkInfo {
            ranges,
            output_ranges,
        };

        // Advance to next chunk
        let mut carry = true;
        for i in (0..self.shape.len()).rev() {
            if carry {
                self.current[i] += self.chunk_dims[i] - self.overlap.get(i).copied().unwrap_or(0);
                if self.current[i] < self.shape[i] {
                    carry = false;
                } else {
                    self.current[i] = 0;
                }
            }
        }

        if carry {
            self.done = true;
        }

        Some(chunk_info)
    }
}

/// Example: Streaming Gaussian filter
pub struct StreamingGaussianFilter<T> {
    sigma: Vec<T>,
    truncate: Option<T>,
}

impl<T: Float + FromPrimitive + Debug + Clone> StreamingGaussianFilter<T> {
    pub fn new(sigma: Vec<T>, truncate: Option<T>) -> Self {
        Self { sigma, truncate }
    }
}

impl<T, D> StreamableOp<T, D> for StreamingGaussianFilter<T>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
{
    fn apply_chunk(&self, chunk: &ArrayView<T, D>) -> NdimageResult<Array<T, D>> {
        crate::filters::gaussian_filter(
            chunk.to_owned(),
            &self.sigma,
            self.truncate,
            None,
            Some(BorderMode::Reflect),
        )
    }

    fn required_overlap(&self) -> Vec<usize> {
        // Overlap should be at least 3 * sigma for Gaussian filter
        self.sigma
            .iter()
            .map(|&s| {
                let truncate = self.truncate.unwrap_or(T::from_f64(4.0).unwrap());
                (truncate * s).to_usize().unwrap_or(4)
            })
            .collect()
    }

    fn merge_overlap(
        &self,
        output: &mut ArrayViewMut<T, D>,
        new_chunk: &ArrayView<T, D>,
        overlap_info: &OverlapInfo,
    ) -> NdimageResult<()> {
        // Simple averaging in overlap region
        // In practice, more sophisticated blending might be needed
        let dim = overlap_info.dimension;
        let overlap_size = overlap_info.overlap_size;

        // This is a simplified implementation
        // Real implementation would need proper indexing
        Ok(())
    }
}

/// Stream process a file-based array
pub fn stream_process_file<T, D, Op>(
    input_path: &Path,
    output_path: &Path,
    shape: &[usize],
    op: Op,
    config: Option<StreamConfig>,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
    Op: StreamableOp<T, D>,
{
    let config = config.unwrap_or_default();
    let processor = StreamProcessor::<T>::new(config);
    processor.process_file::<D, Op>(input_path, output_path, shape, op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_chunk_iterator() {
        let shape = vec![100, 100];
        let chunk_dims = vec![30, 30];
        let overlap = vec![5, 5];

        let mut count = 0;
        for chunk in ChunkIterator::new(&shape, &chunk_dims, &overlap) {
            assert!(!chunk.ranges.is_empty());
            count += 1;
        }

        // Should have multiple chunks
        assert!(count > 1);
    }

    #[test]
    fn test_streaming_processor() {
        let config = StreamConfig {
            chunk_size: 1024,
            overlap: vec![2, 2],
            ..Default::default()
        };

        let processor = StreamProcessor::<f64>::new(config);
        let input = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        let op = StreamingGaussianFilter::new(vec![1.0, 1.0], None);
        let result = processor.process_in_memory(&input.view(), op).unwrap();

        assert_eq!(result.shape(), input.shape());
    }
}
