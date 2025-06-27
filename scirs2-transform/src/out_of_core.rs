//! Out-of-core processing for large datasets
//!
//! This module provides utilities for processing datasets that are too large
//! to fit in memory, using chunked processing and memory-mapped files.

use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix2};
use num_traits::{Float, NumCast};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, Seek, SeekFrom};
use std::path::Path;

use crate::error::{Result, TransformError};
use crate::normalize::NormalizationMethod;

/// Configuration for out-of-core processing
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Maximum chunk size in MB
    pub chunk_size_mb: usize,
    /// Whether to use memory mapping when possible
    pub use_mmap: bool,
    /// Number of threads for parallel processing
    pub n_threads: usize,
    /// Temporary directory for intermediate files
    pub temp_dir: String,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        OutOfCoreConfig {
            chunk_size_mb: 100,
            use_mmap: true,
            n_threads: num_cpus::get(),
            temp_dir: std::env::temp_dir().to_string_lossy().to_string(),
        }
    }
}

/// Trait for transformers that support out-of-core processing
pub trait OutOfCoreTransformer: Send + Sync {
    /// Fit the transformer on chunks of data
    fn fit_chunks<I>(&mut self, chunks: I) -> Result<()>
    where
        I: Iterator<Item = Result<Array2<f64>>>;
    
    /// Transform data in chunks
    fn transform_chunks<I>(&self, chunks: I) -> Result<ChunkedArrayWriter>
    where
        I: Iterator<Item = Result<Array2<f64>>>;
    
    /// Get the expected shape of transformed data
    fn get_transform_shape(&self, input_shape: (usize, usize)) -> (usize, usize);
}

/// Reader for chunked array data from disk
pub struct ChunkedArrayReader {
    file: BufReader<File>,
    shape: (usize, usize),
    chunk_size: usize,
    current_row: usize,
    dtype_size: usize,
}

impl ChunkedArrayReader {
    /// Create a new chunked array reader
    pub fn new<P: AsRef<Path>>(path: P, shape: (usize, usize), chunk_size: usize) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            TransformError::IOError(format!("Failed to open file: {}", e))
        })?;
        
        Ok(ChunkedArrayReader {
            file: BufReader::new(file),
            shape,
            chunk_size,
            current_row: 0,
            dtype_size: std::mem::size_of::<f64>(),
        })
    }
    
    /// Read the next chunk of data
    pub fn read_chunk(&mut self) -> Result<Option<Array2<f64>>> {
        if self.current_row >= self.shape.0 {
            return Ok(None);
        }
        
        let rows_to_read = (self.chunk_size).min(self.shape.0 - self.current_row);
        let mut chunk = Array2::zeros((rows_to_read, self.shape.1));
        
        // Read data row by row
        for i in 0..rows_to_read {
            for j in 0..self.shape.1 {
                let mut bytes = vec![0u8; self.dtype_size];
                self.file.read_exact(&mut bytes).map_err(|e| {
                    TransformError::IOError(format!("Failed to read data: {}", e))
                })?;
                
                chunk[[i, j]] = f64::from_le_bytes(bytes.try_into().unwrap());
            }
        }
        
        self.current_row += rows_to_read;
        Ok(Some(chunk))
    }
    
    /// Create an iterator over chunks
    pub fn chunks(self) -> ChunkedArrayIterator {
        ChunkedArrayIterator { reader: self }
    }
}

/// Iterator over chunks of array data
pub struct ChunkedArrayIterator {
    reader: ChunkedArrayReader,
}

impl Iterator for ChunkedArrayIterator {
    type Item = Result<Array2<f64>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Writer for chunked array data to disk
pub struct ChunkedArrayWriter {
    file: BufWriter<File>,
    shape: (usize, usize),
    rows_written: usize,
    path: String,
}

impl ChunkedArrayWriter {
    /// Create a new chunked array writer
    pub fn new<P: AsRef<Path>>(path: P, shape: (usize, usize)) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::create(&path).map_err(|e| {
            TransformError::IOError(format!("Failed to create file: {}", e))
        })?;
        
        Ok(ChunkedArrayWriter {
            file: BufWriter::new(file),
            shape,
            rows_written: 0,
            path: path_str,
        })
    }
    
    /// Write a chunk of data
    pub fn write_chunk(&mut self, chunk: &Array2<f64>) -> Result<()> {
        if chunk.shape()[1] != self.shape.1 {
            return Err(TransformError::InvalidInput(
                format!("Chunk has {} columns, expected {}", chunk.shape()[1], self.shape.1)
            ));
        }
        
        if self.rows_written + chunk.shape()[0] > self.shape.0 {
            return Err(TransformError::InvalidInput(
                "Too many rows written".to_string()
            ));
        }
        
        // Write data row by row
        for i in 0..chunk.shape()[0] {
            for j in 0..chunk.shape()[1] {
                let bytes = chunk[[i, j]].to_le_bytes();
                self.file.write_all(&bytes).map_err(|e| {
                    TransformError::IOError(format!("Failed to write data: {}", e))
                })?;
            }
        }
        
        self.rows_written += chunk.shape()[0];
        Ok(())
    }
    
    /// Finalize the writer and flush data
    pub fn finalize(mut self) -> Result<String> {
        self.file.flush().map_err(|e| {
            TransformError::IOError(format!("Failed to flush data: {}", e))
        })?;
        
        if self.rows_written != self.shape.0 {
            return Err(TransformError::InvalidInput(
                format!("Expected {} rows, but wrote {}", self.shape.0, self.rows_written)
            ));
        }
        
        Ok(self.path)
    }
}

/// Out-of-core normalizer implementation
pub struct OutOfCoreNormalizer {
    method: NormalizationMethod,
    axis: usize,
    // Statistics computed during fit
    stats: Option<NormalizationStats>,
}

#[derive(Clone)]
struct NormalizationStats {
    min: Array1<f64>,
    max: Array1<f64>,
    mean: Array1<f64>,
    std: Array1<f64>,
    median: Array1<f64>,
    iqr: Array1<f64>,
    count: usize,
}

impl OutOfCoreNormalizer {
    /// Create a new out-of-core normalizer
    pub fn new(method: NormalizationMethod, axis: usize) -> Self {
        OutOfCoreNormalizer {
            method,
            axis,
            stats: None,
        }
    }
    
    /// Compute statistics in a single pass for simple methods
    fn compute_simple_stats<I>(&mut self, chunks: I, n_features: usize) -> Result<()>
    where
        I: Iterator<Item = Result<Array2<f64>>>,
    {
        let mut min = Array1::from_elem(n_features, f64::INFINITY);
        let mut max = Array1::from_elem(n_features, f64::NEG_INFINITY);
        let mut sum = Array1::zeros(n_features);
        let mut sum_sq = Array1::zeros(n_features);
        let mut count = 0;
        
        // First pass: compute min, max, sum, sum_sq
        for chunk_result in chunks {
            let chunk = chunk_result?;
            count += chunk.shape()[0];
            
            for j in 0..n_features {
                let col = chunk.column(j);
                for &val in col.iter() {
                    min[j] = min[j].min(val);
                    max[j] = max[j].max(val);
                    sum[j] += val;
                    sum_sq[j] += val * val;
                }
            }
        }
        
        // Compute mean and std
        let mean = sum / count as f64;
        let variance = sum_sq / count as f64 - &mean * &mean;
        let std = variance.mapv(|v| v.sqrt());
        
        self.stats = Some(NormalizationStats {
            min,
            max,
            mean,
            std,
            median: Array1::zeros(n_features), // Not used for simple methods
            iqr: Array1::zeros(n_features),    // Not used for simple methods
            count,
        });
        
        Ok(())
    }
}

impl OutOfCoreTransformer for OutOfCoreNormalizer {
    fn fit_chunks<I>(&mut self, chunks: I) -> Result<()>
    where
        I: Iterator<Item = Result<Array2<f64>>>,
    {
        // Peek at the first chunk to get dimensions
        let mut chunks_iter = chunks.peekable();
        let n_features = match chunks_iter.peek() {
            Some(Ok(chunk)) => chunk.shape()[1],
            Some(Err(_)) => return chunks_iter.next().unwrap().map(|_| ()),
            None => return Err(TransformError::InvalidInput("No chunks provided".to_string())),
        };
        
        match self.method {
            NormalizationMethod::MinMax | 
            NormalizationMethod::MinMaxCustom(_, _) |
            NormalizationMethod::ZScore |
            NormalizationMethod::MaxAbs => {
                self.compute_simple_stats(chunks_iter, n_features)?;
            }
            NormalizationMethod::Robust => {
                // Robust scaling requires multiple passes or approximate methods
                // For now, we'll use a reservoir sampling approach
                return Err(TransformError::NotImplemented(
                    "Robust scaling not yet implemented for out-of-core processing".to_string()
                ));
            }
            _ => {
                return Err(TransformError::NotImplemented(
                    "This normalization method is not supported for out-of-core processing".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn transform_chunks<I>(&self, chunks: I) -> Result<ChunkedArrayWriter>
    where
        I: Iterator<Item = Result<Array2<f64>>>,
    {
        if self.stats.is_none() {
            return Err(TransformError::TransformationError(
                "Normalizer has not been fitted".to_string()
            ));
        }
        
        let stats = self.stats.as_ref().unwrap();
        
        // Create temporary output file
        let output_path = format!("{}/transform_output_{}.bin", 
                                  std::env::temp_dir().to_string_lossy(),
                                  std::process::id());
        
        let mut writer = ChunkedArrayWriter::new(
            &output_path,
            (stats.count, stats.min.len())
        )?;
        
        // Transform each chunk
        for chunk_result in chunks {
            let chunk = chunk_result?;
            let mut transformed = Array2::zeros(chunk.shape());
            
            match self.method {
                NormalizationMethod::MinMax => {
                    let range = &stats.max - &stats.min;
                    for i in 0..chunk.shape()[0] {
                        for j in 0..chunk.shape()[1] {
                            if range[j].abs() > 1e-10 {
                                transformed[[i, j]] = (chunk[[i, j]] - stats.min[j]) / range[j];
                            } else {
                                transformed[[i, j]] = 0.5;
                            }
                        }
                    }
                }
                NormalizationMethod::ZScore => {
                    for i in 0..chunk.shape()[0] {
                        for j in 0..chunk.shape()[1] {
                            if stats.std[j] > 1e-10 {
                                transformed[[i, j]] = (chunk[[i, j]] - stats.mean[j]) / stats.std[j];
                            } else {
                                transformed[[i, j]] = 0.0;
                            }
                        }
                    }
                }
                NormalizationMethod::MaxAbs => {
                    for i in 0..chunk.shape()[0] {
                        for j in 0..chunk.shape()[1] {
                            let max_abs = stats.max[j].abs().max(stats.min[j].abs());
                            if max_abs > 1e-10 {
                                transformed[[i, j]] = chunk[[i, j]] / max_abs;
                            } else {
                                transformed[[i, j]] = 0.0;
                            }
                        }
                    }
                }
                _ => {
                    return Err(TransformError::NotImplemented(
                        "This normalization method is not supported".to_string()
                    ));
                }
            }
            
            writer.write_chunk(&transformed)?;
        }
        
        Ok(writer)
    }
    
    fn get_transform_shape(&self, input_shape: (usize, usize)) -> (usize, usize) {
        input_shape // Normalization doesn't change shape
    }
}

/// Create chunks from a large CSV file
pub fn csv_chunks<P: AsRef<Path>>(
    path: P,
    chunk_size: usize,
    has_header: bool,
) -> Result<impl Iterator<Item = Result<Array2<f64>>>> {
    let file = File::open(path).map_err(|e| {
        TransformError::IOError(format!("Failed to open CSV file: {}", e))
    })?;
    
    Ok(CsvChunkIterator::new(BufReader::new(file), chunk_size, has_header))
}

/// Iterator that reads CSV in chunks
struct CsvChunkIterator {
    reader: BufReader<File>,
    chunk_size: usize,
    skip_header: bool,
    header_skipped: bool,
}

impl CsvChunkIterator {
    fn new(reader: BufReader<File>, chunk_size: usize, skip_header: bool) -> Self {
        CsvChunkIterator {
            reader,
            chunk_size,
            skip_header,
            header_skipped: false,
        }
    }
}

impl Iterator for CsvChunkIterator {
    type Item = Result<Array2<f64>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        use std::io::BufRead;
        
        let mut rows = Vec::new();
        let mut n_cols = None;
        
        for line_result in (&mut self.reader).lines().take(self.chunk_size) {
            let line = match line_result {
                Ok(l) => l,
                Err(e) => return Some(Err(TransformError::IOError(
                    format!("Failed to read line: {}", e)
                ))),
            };
            
            // Skip header if needed
            if self.skip_header && !self.header_skipped {
                self.header_skipped = true;
                continue;
            }
            
            // Parse CSV line
            let values: Result<Vec<f64>> = line
                .split(',')
                .map(|s| s.trim().parse::<f64>()
                    .map_err(|e| TransformError::ParseError(
                        format!("Failed to parse number: {}", e)
                    )))
                .collect();
            
            let values = match values {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };
            
            // Check column consistency
            if let Some(nc) = n_cols {
                if values.len() != nc {
                    return Some(Err(TransformError::InvalidInput(
                        "Inconsistent number of columns in CSV".to_string()
                    )));
                }
            } else {
                n_cols = Some(values.len());
            }
            
            rows.push(values);
        }
        
        if rows.is_empty() {
            return None;
        }
        
        // Convert to Array2
        let n_rows = rows.len();
        let n_cols = n_cols.unwrap();
        let mut array = Array2::zeros((n_rows, n_cols));
        
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                array[[i, j]] = val;
            }
        }
        
        Some(Ok(array))
    }
}