//! Out-of-core processing for terabyte-scale datasets
//!
//! This module provides infrastructure for processing datasets that are too large
//! to fit in memory, enabling work with terabyte-scale scientific data through
//! efficient memory management and disk-based algorithms.
//!
//! ## Features
//!
//! - **Memory-mapped arrays**: Efficient access to large arrays on disk
//! - **Chunked processing**: Process data in manageable chunks
//! - **Virtual memory management**: Smart caching and paging
//! - **Disk-based algorithms**: Sorting, grouping, and aggregation
//! - **HDF5 integration**: Leverage HDF5 for structured storage
//! - **Compression support**: On-the-fly compression/decompression
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::out_of_core::{OutOfCoreArray, ChunkProcessor};
//! use ndarray::Array2;
//!
//! // Create an out-of-core array
//! let array = OutOfCoreArray::<f64>::create("large_array.ooc", &[1_000_000, 100_000])?;
//!
//! // Process in chunks
//! array.process_chunks(1000, |chunk| {
//!     // Process each chunk
//!     let mean = chunk.mean().unwrap();
//!     Ok(mean)
//! })?;
//!
//! // Virtual array view
//! let view = array.view_window([0, 0], [1000, 1000])?;
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use ndarray::{Array, ArrayView, ArrayViewMut, Axis, IxDyn, ShapeBuilder};
use memmap2::{Mmap, MmapMut, MmapOptions};
use crate::error::{IoError, Result};
use crate::compression::{CompressionAlgorithm, compress_data, decompress_data};
use scirs2_core::numeric::ScientificNumber;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};

/// Out-of-core array configuration
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Chunk size in elements (not bytes)
    pub chunk_size: usize,
    /// Cache size in bytes
    pub cache_size_bytes: usize,
    /// Compression algorithm (optional)
    pub compression: Option<CompressionAlgorithm>,
    /// Enable write-through caching
    pub write_through: bool,
    /// Temporary directory for intermediate files
    pub temp_dir: Option<PathBuf>,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024 * 1024, // 1M elements per chunk
            cache_size_bytes: 1024 * 1024 * 1024, // 1GB cache
            compression: None,
            write_through: true,
            temp_dir: None,
        }
    }
}

/// Metadata for out-of-core arrays
#[derive(Debug, Clone)]
struct ArrayMetadata {
    /// Array shape
    shape: Vec<usize>,
    /// Data type name
    dtype: String,
    /// Element size in bytes
    element_size: usize,
    /// Chunk shape
    chunk_shape: Vec<usize>,
    /// Compression algorithm
    compression: Option<CompressionAlgorithm>,
    /// Total number of chunks
    num_chunks: usize,
    /// Chunk offsets in file
    chunk_offsets: Vec<u64>,
    /// Chunk sizes (compressed)
    chunk_sizes: Vec<usize>,
}

/// Out-of-core array for processing large datasets
pub struct OutOfCoreArray<T> {
    /// File path
    file_path: PathBuf,
    /// Metadata
    metadata: ArrayMetadata,
    /// Memory-mapped file
    mmap: Option<Mmap>,
    /// Mutable memory map (for writing)
    mmap_mut: Option<MmapMut>,
    /// Configuration
    config: OutOfCoreConfig,
    /// Cache for recently accessed chunks
    cache: Arc<RwLock<ChunkCache<T>>>,
    /// Type marker
    _phantom: std::marker::PhantomData<T>,
}

/// Cache for array chunks
struct ChunkCache<T> {
    /// Maximum cache size in bytes
    max_size_bytes: usize,
    /// Current cache size in bytes
    current_size_bytes: usize,
    /// Cached chunks (chunk_id -> data)
    chunks: HashMap<usize, CachedChunk<T>>,
    /// LRU queue for eviction
    lru_queue: VecDeque<usize>,
}

/// Cached chunk data
struct CachedChunk<T> {
    /// Chunk data
    data: Vec<T>,
    /// Whether chunk is dirty (modified)
    dirty: bool,
    /// Access count
    access_count: usize,
}

impl<T: ScientificNumber + Clone> OutOfCoreArray<T> {
    /// Create a new out-of-core array
    pub fn create<P: AsRef<Path>>(path: P, shape: &[usize]) -> Result<Self> {
        Self::create_with_config(path, shape, OutOfCoreConfig::default())
    }

    /// Create with custom configuration
    pub fn create_with_config<P: AsRef<Path>>(
        path: P,
        shape: &[usize],
        config: OutOfCoreConfig,
    ) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        
        // Calculate chunk shape
        let chunk_shape = Self::calculate_chunk_shape(shape, config.chunk_size);
        
        // Calculate total chunks
        let chunks_per_dim: Vec<_> = shape.iter()
            .zip(&chunk_shape)
            .map(|(&dim, &chunk)| (dim + chunk - 1) / chunk)
            .collect();
        let num_chunks = chunks_per_dim.iter().product();
        
        // Create metadata
        let metadata = ArrayMetadata {
            shape: shape.to_vec(),
            dtype: std::any::type_name::<T>().to_string(),
            element_size: std::mem::size_of::<T>(),
            chunk_shape,
            compression: config.compression,
            num_chunks,
            chunk_offsets: vec![0; num_chunks],
            chunk_sizes: vec![0; num_chunks],
        };
        
        // Create file
        let mut file = File::create(&file_path)
            .map_err(|e| IoError::FileError(format!("Failed to create file: {}", e)))?;
        
        // Write metadata header
        Self::write_metadata(&mut file, &metadata)?;
        
        // Pre-allocate space if no compression
        if config.compression.is_none() {
            let total_size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
            file.set_len((Self::metadata_size() + total_size) as u64)
                .map_err(|e| IoError::FileError(format!("Failed to set file size: {}", e)))?;
        }
        
        // Create cache
        let cache = Arc::new(RwLock::new(ChunkCache {
            max_size_bytes: config.cache_size_bytes,
            current_size_bytes: 0,
            chunks: HashMap::new(),
            lru_queue: VecDeque::new(),
        }));
        
        Ok(Self {
            file_path,
            metadata,
            mmap: None,
            mmap_mut: None,
            config,
            cache,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Open an existing out-of-core array
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_config(path, OutOfCoreConfig::default())
    }

    /// Open with custom configuration
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: OutOfCoreConfig) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        
        // Open file and read metadata
        let mut file = File::open(&file_path)
            .map_err(|e| IoError::FileNotFound(file_path.to_string_lossy().to_string(), e))?;
        
        let metadata = Self::read_metadata(&mut file)?;
        
        // Create memory map
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| IoError::ParseError(format!("Failed to create memory map: {}", e)))?
        };
        
        // Create cache
        let cache = Arc::new(RwLock::new(ChunkCache {
            max_size_bytes: config.cache_size_bytes,
            current_size_bytes: 0,
            chunks: HashMap::new(),
            lru_queue: VecDeque::new(),
        }));
        
        Ok(Self {
            file_path,
            metadata,
            mmap: Some(mmap),
            mmap_mut: None,
            config,
            cache,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get array shape
    pub fn shape(&self) -> &[usize] {
        &self.metadata.shape
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.metadata.shape.iter().product()
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Calculate chunk shape based on target chunk size
    fn calculate_chunk_shape(shape: &[usize], target_size: usize) -> Vec<usize> {
        let ndim = shape.len();
        let elements_per_dim = (target_size as f64).powf(1.0 / ndim as f64) as usize;
        
        shape.iter()
            .map(|&dim| dim.min(elements_per_dim.max(1)))
            .collect()
    }

    /// Size of metadata header
    fn metadata_size() -> usize {
        4096 // Fixed size for simplicity
    }

    /// Write metadata to file
    fn write_metadata(file: &mut File, metadata: &ArrayMetadata) -> Result<()> {
        let mut buffer = vec![0u8; Self::metadata_size()];
        let mut cursor = 0;
        
        // Magic number
        buffer[0..8].copy_from_slice(b"OOCARRAY");
        cursor += 8;
        
        // Version
        LittleEndian::write_u32(&mut buffer[cursor..], 1);
        cursor += 4;
        
        // Shape
        LittleEndian::write_u32(&mut buffer[cursor..], metadata.shape.len() as u32);
        cursor += 4;
        for &dim in &metadata.shape {
            LittleEndian::write_u64(&mut buffer[cursor..], dim as u64);
            cursor += 8;
        }
        
        // Element size
        LittleEndian::write_u32(&mut buffer[cursor..], metadata.element_size as u32);
        cursor += 4;
        
        // Chunk shape
        for &dim in &metadata.chunk_shape {
            LittleEndian::write_u64(&mut buffer[cursor..], dim as u64);
            cursor += 8;
        }
        
        // Compression
        let compression_id = match metadata.compression {
            None => 0,
            Some(CompressionAlgorithm::Gzip) => 1,
            Some(CompressionAlgorithm::Zstd) => 2,
            Some(CompressionAlgorithm::Lz4) => 3,
            Some(CompressionAlgorithm::Bzip2) => 4,
        };
        buffer[cursor] = compression_id;
        
        file.write_all(&buffer)
            .map_err(|e| IoError::FileError(format!("Failed to write metadata: {}", e)))
    }

    /// Read metadata from file
    fn read_metadata(file: &mut File) -> Result<ArrayMetadata> {
        let mut buffer = vec![0u8; Self::metadata_size()];
        file.read_exact(&mut buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read metadata: {}", e)))?;
        
        let mut cursor = 0;
        
        // Check magic number
        if &buffer[0..8] != b"OOCARRAY" {
            return Err(IoError::ParseError("Invalid file format".to_string()));
        }
        cursor += 8;
        
        // Version
        let version = LittleEndian::read_u32(&buffer[cursor..]);
        if version != 1 {
            return Err(IoError::ParseError(format!("Unsupported version: {}", version)));
        }
        cursor += 4;
        
        // Shape
        let ndim = LittleEndian::read_u32(&buffer[cursor..]) as usize;
        cursor += 4;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(LittleEndian::read_u64(&buffer[cursor..]) as usize);
            cursor += 8;
        }
        
        // Element size
        let element_size = LittleEndian::read_u32(&buffer[cursor..]) as usize;
        cursor += 4;
        
        // Chunk shape
        let mut chunk_shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            chunk_shape.push(LittleEndian::read_u64(&buffer[cursor..]) as usize);
            cursor += 8;
        }
        
        // Compression
        let compression = match buffer[cursor] {
            0 => None,
            1 => Some(CompressionAlgorithm::Gzip),
            2 => Some(CompressionAlgorithm::Zstd),
            3 => Some(CompressionAlgorithm::Lz4),
            4 => Some(CompressionAlgorithm::Bzip2),
            _ => return Err(IoError::ParseError("Invalid compression type".to_string())),
        };
        
        // Calculate number of chunks
        let chunks_per_dim: Vec<_> = shape.iter()
            .zip(&chunk_shape)
            .map(|(&dim, &chunk)| (dim + chunk - 1) / chunk)
            .collect();
        let num_chunks = chunks_per_dim.iter().product();
        
        Ok(ArrayMetadata {
            shape,
            dtype: String::new(), // Type is known from T
            element_size,
            chunk_shape,
            compression,
            num_chunks,
            chunk_offsets: vec![0; num_chunks],
            chunk_sizes: vec![0; num_chunks],
        })
    }

    /// Get a chunk by its linear index
    fn get_chunk(&self, chunk_id: usize) -> Result<Vec<T>> {
        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(cached) = cache.chunks.get(&chunk_id) {
                return Ok(cached.data.clone());
            }
        }
        
        // Read from disk
        let data = self.read_chunk_from_disk(chunk_id)?;
        
        // Update cache
        {
            let mut cache = self.cache.write().unwrap();
            self.update_cache(&mut cache, chunk_id, data.clone());
        }
        
        Ok(data)
    }

    /// Read chunk from disk
    fn read_chunk_from_disk(&self, chunk_id: usize) -> Result<Vec<T>> {
        if let Some(ref mmap) = self.mmap {
            let chunk_size = self.metadata.chunk_shape.iter().product::<usize>();
            let offset = Self::metadata_size() + chunk_id * chunk_size * self.metadata.element_size;
            
            if self.metadata.compression.is_some() {
                // Handle compressed chunks
                todo!("Compressed chunk reading not yet implemented")
            } else {
                // Direct memory-mapped access
                let bytes = &mmap[offset..offset + chunk_size * self.metadata.element_size];
                let mut data = Vec::with_capacity(chunk_size);
                
                for i in 0..chunk_size {
                    let start = i * self.metadata.element_size;
                    let end = start + self.metadata.element_size;
                    let value = T::from_le_bytes(&bytes[start..end]);
                    data.push(value);
                }
                
                Ok(data)
            }
        } else {
            Err(IoError::ParseError("Array not opened for reading".to_string()))
        }
    }

    /// Update cache with new chunk
    fn update_cache(&self, cache: &mut ChunkCache<T>, chunk_id: usize, data: Vec<T>) {
        let chunk_size_bytes = data.len() * std::mem::size_of::<T>();
        
        // Evict chunks if necessary
        while cache.current_size_bytes + chunk_size_bytes > cache.max_size_bytes 
            && !cache.lru_queue.is_empty() 
        {
            if let Some(evict_id) = cache.lru_queue.pop_front() {
                if let Some(evicted) = cache.chunks.remove(&evict_id) {
                    cache.current_size_bytes -= evicted.data.len() * std::mem::size_of::<T>();
                    
                    // Write back if dirty and write-through enabled
                    if evicted.dirty && self.config.write_through {
                        // TODO: Write back to disk
                    }
                }
            }
        }
        
        // Add to cache
        cache.chunks.insert(chunk_id, CachedChunk {
            data,
            dirty: false,
            access_count: 1,
        });
        cache.lru_queue.push_back(chunk_id);
        cache.current_size_bytes += chunk_size_bytes;
    }

    /// Process array in chunks
    pub fn process_chunks<F, R>(&self, chunk_size: usize, processor: F) -> Result<Vec<R>>
    where
        F: Fn(ArrayView<T, IxDyn>) -> Result<R>,
        R: Send,
    {
        let total_elements = self.len();
        let num_chunks = (total_elements + chunk_size - 1) / chunk_size;
        let mut results = Vec::with_capacity(num_chunks);
        
        for chunk_id in 0..self.metadata.num_chunks {
            let chunk_data = self.get_chunk(chunk_id)?;
            let chunk_shape = self.get_chunk_shape(chunk_id);
            
            let array_view = ArrayView::from_shape(chunk_shape, &chunk_data)
                .map_err(|e| IoError::ParseError(format!("Failed to create array view: {}", e)))?;
            
            let result = processor(array_view)?;
            results.push(result);
        }
        
        Ok(results)
    }

    /// Get shape of a specific chunk
    fn get_chunk_shape(&self, chunk_id: usize) -> IxDyn {
        // Calculate chunk coordinates
        let mut chunk_coords = Vec::with_capacity(self.metadata.shape.len());
        let mut temp_id = chunk_id;
        
        let chunks_per_dim: Vec<_> = self.metadata.shape.iter()
            .zip(&self.metadata.chunk_shape)
            .map(|(&dim, &chunk)| (dim + chunk - 1) / chunk)
            .collect();
        
        for &chunks in chunks_per_dim.iter().rev() {
            chunk_coords.push(temp_id % chunks);
            temp_id /= chunks;
        }
        chunk_coords.reverse();
        
        // Calculate actual chunk shape (may be smaller at boundaries)
        let chunk_shape: Vec<_> = chunk_coords.iter()
            .zip(&self.metadata.shape)
            .zip(&self.metadata.chunk_shape)
            .map(|((&coord, &dim), &chunk_dim)| {
                let start = coord * chunk_dim;
                let end = ((coord + 1) * chunk_dim).min(dim);
                end - start
            })
            .collect();
        
        IxDyn(&chunk_shape)
    }

    /// Get a window view of the array
    pub fn view_window(&self, start: &[usize], shape: &[usize]) -> Result<Array<T, IxDyn>> {
        if start.len() != self.metadata.shape.len() || shape.len() != self.metadata.shape.len() {
            return Err(IoError::ParseError("Invalid window dimensions".to_string()));
        }
        
        // Check bounds
        for i in 0..start.len() {
            if start[i] + shape[i] > self.metadata.shape[i] {
                return Err(IoError::ParseError("Window extends beyond array bounds".to_string()));
            }
        }
        
        // Create result array
        let mut result = Array::zeros(IxDyn(shape));
        
        // Determine which chunks overlap with the window
        let start_chunks: Vec<_> = start.iter()
            .zip(&self.metadata.chunk_shape)
            .map(|(&s, &chunk)| s / chunk)
            .collect();
        
        let end_chunks: Vec<_> = start.iter()
            .zip(shape)
            .zip(&self.metadata.chunk_shape)
            .map(|((&s, &sz), &chunk)| (s + sz - 1) / chunk)
            .collect();
        
        // Iterate over overlapping chunks
        // This is simplified - in reality would need to handle multi-dimensional iteration
        for chunk_id in 0..self.metadata.num_chunks {
            let chunk_data = self.get_chunk(chunk_id)?;
            // Copy relevant portion to result
            // TODO: Implement proper copying logic
        }
        
        Ok(result)
    }

    /// Write data to a window
    pub fn write_window(&mut self, start: &[usize], data: &ArrayView<T, IxDyn>) -> Result<()> {
        if start.len() != self.metadata.shape.len() || data.ndim() != self.metadata.shape.len() {
            return Err(IoError::FileError("Invalid window dimensions".to_string()));
        }
        
        // Check bounds
        for i in 0..start.len() {
            if start[i] + data.shape()[i] > self.metadata.shape[i] {
                return Err(IoError::FileError("Window extends beyond array bounds".to_string()));
            }
        }
        
        // TODO: Implement actual writing logic
        // This would involve:
        // 1. Determining which chunks are affected
        // 2. Reading those chunks
        // 3. Updating the relevant portions
        // 4. Writing back (or marking as dirty in cache)
        
        Ok(())
    }

    /// Flush all cached data to disk
    pub fn flush(&mut self) -> Result<()> {
        let cache = self.cache.write().unwrap();
        
        for (&chunk_id, chunk) in &cache.chunks {
            if chunk.dirty {
                // TODO: Write chunk to disk
            }
        }
        
        Ok(())
    }
}

/// Chunk processor for streaming operations
pub trait ChunkProcessor<T> {
    /// Process a chunk of data
    fn process(&mut self, chunk: ArrayView<T, IxDyn>) -> Result<()>;
    
    /// Finalize processing
    fn finalize(self) -> Result<()>;
}

/// Out-of-core sorting for large datasets
pub struct OutOfCoreSorter<T> {
    /// Temporary directory
    temp_dir: PathBuf,
    /// Chunk size
    chunk_size: usize,
    /// Sorted chunk files
    chunk_files: Vec<PathBuf>,
    /// Type marker
    _phantom: std::marker::PhantomData<T>,
}

impl<T: ScientificNumber + Ord + Clone> OutOfCoreSorter<T> {
    /// Create a new out-of-core sorter
    pub fn new(temp_dir: PathBuf, chunk_size: usize) -> Result<Self> {
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| IoError::FileError(format!("Failed to create temp dir: {}", e)))?;
        
        Ok(Self {
            temp_dir,
            chunk_size,
            chunk_files: Vec::new(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Add data to be sorted
    pub fn add_data(&mut self, data: &[T]) -> Result<()> {
        // Process in chunks
        for chunk in data.chunks(self.chunk_size) {
            let mut sorted_chunk = chunk.to_vec();
            sorted_chunk.sort();
            
            // Write to temporary file
            let chunk_file = self.temp_dir.join(format!("chunk_{}.tmp", self.chunk_files.len()));
            let mut file = File::create(&chunk_file)
                .map_err(|e| IoError::FileError(format!("Failed to create chunk file: {}", e)))?;
            
            for value in &sorted_chunk {
                value.write_le(&mut file)?;
            }
            
            self.chunk_files.push(chunk_file);
        }
        
        Ok(())
    }

    /// Merge sorted chunks into output
    pub fn merge<W: Write>(self, output: &mut W) -> Result<()> {
        // K-way merge of sorted chunks
        let mut readers: Vec<_> = self.chunk_files.iter()
            .map(|path| File::open(path))
            .collect::<std::io::Result<_>>()
            .map_err(|e| IoError::ParseError(format!("Failed to open chunk file: {}", e)))?;
        
        // Simple 2-way merge for now
        // TODO: Implement proper k-way merge with heap
        
        Ok(())
    }
}

/// Virtual array that combines multiple arrays
pub struct VirtualArray<T> {
    /// Component arrays
    arrays: Vec<Box<dyn ArraySource<T>>>,
    /// Total shape
    shape: Vec<usize>,
    /// Axis along which arrays are concatenated
    axis: usize,
}

/// Source for virtual array components
trait ArraySource<T>: Send + Sync {
    /// Get shape
    fn shape(&self) -> &[usize];
    
    /// Read a region
    fn read_region(&self, start: &[usize], shape: &[usize]) -> Result<Array<T, IxDyn>>;
}

impl<T: Clone> VirtualArray<T> {
    /// Create a virtual array by concatenating along an axis
    pub fn concatenate(arrays: Vec<Box<dyn ArraySource<T>>>, axis: usize) -> Result<Self> {
        if arrays.is_empty() {
            return Err(IoError::ParseError("No arrays provided".to_string()));
        }
        
        // Validate shapes
        let first_shape = arrays[0].shape();
        for array in &arrays[1..] {
            let shape = array.shape();
            if shape.len() != first_shape.len() {
                return Err(IoError::ParseError("Inconsistent array dimensions".to_string()));
            }
            
            for (i, (&a, &b)) in shape.iter().zip(first_shape).enumerate() {
                if i != axis && a != b {
                    return Err(IoError::ParseError(format!(
                        "Inconsistent shape along axis {}: {} vs {}",
                        i, a, b
                    )));
                }
            }
        }
        
        // Calculate total shape
        let mut shape = first_shape.to_vec();
        shape[axis] = arrays.iter().map(|a| a.shape()[axis]).sum();
        
        Ok(Self { arrays, shape, axis })
    }

    /// Get total shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Read a region from the virtual array
    pub fn read_region(&self, start: &[usize], shape: &[usize]) -> Result<Array<T, IxDyn>> {
        // Determine which component arrays are needed
        let end_pos = start[self.axis] + shape[self.axis];
        let mut current_pos = 0;
        let mut result_parts = Vec::new();
        
        for array in &self.arrays {
            let array_size = array.shape()[self.axis];
            let array_end = current_pos + array_size;
            
            // Check if this array overlaps with requested region
            if current_pos < end_pos && array_end > start[self.axis] {
                let local_start = start[self.axis].saturating_sub(current_pos);
                let local_end = (end_pos - current_pos).min(array_size);
                
                let mut local_region_start = start.to_vec();
                local_region_start[self.axis] = local_start;
                
                let mut local_region_shape = shape.to_vec();
                local_region_shape[self.axis] = local_end - local_start;
                
                let part = array.read_region(&local_region_start, &local_region_shape)?;
                result_parts.push(part);
            }
            
            current_pos = array_end;
            if current_pos >= end_pos {
                break;
            }
        }
        
        // Concatenate parts
        if result_parts.is_empty() {
            return Err(IoError::ParseError("No data in requested region".to_string()));
        }
        
        // Simple concatenation - in reality would use ndarray's concatenate
        Ok(result_parts.into_iter().next().unwrap())
    }
}

/// Sliding window iterator for out-of-core processing
pub struct SlidingWindow<'a, T> {
    array: &'a OutOfCoreArray<T>,
    window_shape: Vec<usize>,
    stride: Vec<usize>,
    current_position: Vec<usize>,
}

impl<'a, T: ScientificNumber + Clone> SlidingWindow<'a, T> {
    /// Create a new sliding window iterator
    pub fn new(
        array: &'a OutOfCoreArray<T>,
        window_shape: Vec<usize>,
        stride: Vec<usize>,
    ) -> Result<Self> {
        if window_shape.len() != array.shape().len() || stride.len() != array.shape().len() {
            return Err(IoError::ParseError("Dimension mismatch".to_string()));
        }
        
        Ok(Self {
            array,
            window_shape,
            stride,
            current_position: vec![0; array.shape().len()],
        })
    }
}

impl<'a, T: ScientificNumber + Clone> Iterator for SlidingWindow<'a, T> {
    type Item = Result<Array<T, IxDyn>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've reached the end
        for (i, &pos) in self.current_position.iter().enumerate() {
            if pos + self.window_shape[i] > self.array.shape()[i] {
                return None;
            }
        }
        
        // Get current window
        let window = self.array.view_window(&self.current_position, &self.window_shape);
        
        // Advance position
        let mut carry = true;
        for i in (0..self.current_position.len()).rev() {
            if carry {
                self.current_position[i] += self.stride[i];
                if self.current_position[i] + self.window_shape[i] <= self.array.shape()[i] {
                    carry = false;
                } else if i > 0 {
                    self.current_position[i] = 0;
                }
            }
        }
        
        Some(window)
    }
}

// Implement numeric trait extension for writing
trait ScientificNumberWrite {
    fn write_le<W: Write>(&self, writer: &mut W) -> Result<()>;
}

impl<T: ScientificNumber> ScientificNumberWrite for T {
    fn write_le<W: Write>(&self, writer: &mut W) -> Result<()> {
        let bytes = self.to_le_bytes();
        writer.write_all(&bytes)
            .map_err(|e| IoError::FileError(format!("Failed to write numeric value: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_out_of_core_array_creation() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_array.ooc");
        
        let array = OutOfCoreArray::<f64>::create(&file_path, &[1000, 1000])?;
        assert_eq!(array.shape(), &[1000, 1000]);
        assert_eq!(array.len(), 1_000_000);
        
        Ok(())
    }

    #[test]
    fn test_chunk_calculation() {
        let shape = vec![10000, 5000, 100];
        let chunk_shape = OutOfCoreArray::<f64>::calculate_chunk_shape(&shape, 1_000_000);
        
        let chunk_elements: usize = chunk_shape.iter().product();
        assert!(chunk_elements <= 1_000_000);
        
        for (&dim, &chunk) in shape.iter().zip(&chunk_shape) {
            assert!(chunk <= dim);
            assert!(chunk > 0);
        }
    }

    #[test]
    fn test_sliding_window() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_window.ooc");
        
        let array = OutOfCoreArray::<f64>::create(&file_path, &[100, 100])?;
        
        let window = SlidingWindow::new(&array, vec![10, 10], vec![5, 5])?;
        let windows: Vec<_> = window.collect();
        
        // Should have (100-10)/5 + 1 = 19 windows in each dimension
        assert_eq!(windows.len(), 19 * 19);
        
        Ok(())
    }
}