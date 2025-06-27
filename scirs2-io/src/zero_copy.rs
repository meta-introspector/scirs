//! Zero-copy I/O optimizations
//!
//! This module provides zero-copy implementations for various I/O operations
//! to minimize memory allocations and improve performance with large datasets.

use crate::error::{IoError, Result};
use memmap2::{Mmap, MmapMut, MmapOptions};
use ndarray::{ArrayView, ArrayViewMut, IxDyn, Array1, ArrayView1};
use std::fs::{File, OpenOptions};
use std::marker::PhantomData;
use std::mem;
use std::path::Path;
use std::slice;
use scirs2_core::simd_ops::{SimdUnifiedOps, PlatformCapabilities};
use scirs2_core::parallel_ops::*;

/// Zero-copy array view over memory-mapped data
pub struct ZeroCopyArrayView<'a, T> {
    mmap: &'a Mmap,
    shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<'a, T> ZeroCopyArrayView<'a, T>
where
    T: 'static + Copy,
{
    /// Apply SIMD operations on the array view
    pub fn apply_simd_operation<F>(&self, op: F) -> Result<Vec<T>>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        let slice = self.as_slice();
        Ok(op(slice))
    }

    /// Create a new zero-copy array view from memory-mapped data
    pub fn new(mmap: &'a Mmap, shape: Vec<usize>) -> Result<Self> {
        let expected_bytes = shape.iter().product::<usize>() * mem::size_of::<T>();
        if mmap.len() < expected_bytes {
            return Err(IoError::FormatError(format!(
                "Memory map too small: expected {} bytes, got {}",
                expected_bytes,
                mmap.len()
            )));
        }
        
        Ok(Self {
            mmap,
            shape,
            _phantom: PhantomData,
        })
    }
    
    /// Get an ndarray view without copying data
    pub fn as_array_view(&self) -> ArrayView<T, IxDyn> {
        let ptr = self.mmap.as_ptr() as *const T;
        let slice = unsafe {
            slice::from_raw_parts(ptr, self.shape.iter().product())
        };
        
        ArrayView::from_shape(IxDyn(&self.shape), slice)
            .expect("Shape mismatch in zero-copy view")
    }
    
    /// Get a slice view of the data
    pub fn as_slice(&self) -> &[T] {
        let ptr = self.mmap.as_ptr() as *const T;
        let len = self.shape.iter().product();
        unsafe { slice::from_raw_parts(ptr, len) }
    }
}

/// Zero-copy mutable array view
pub struct ZeroCopyArrayViewMut<'a, T> {
    mmap: &'a mut MmapMut,
    shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<'a, T> ZeroCopyArrayViewMut<'a, T>
where
    T: 'static + Copy,
{
    /// Apply SIMD operations in-place on the mutable array view
    pub fn apply_simd_operation_inplace<F>(&mut self, op: F) -> Result<()>
    where
        F: Fn(&mut [T]),
    {
        let slice = self.as_slice_mut();
        op(slice);
        Ok(())
    }

    /// Create a new mutable zero-copy array view
    pub fn new(mmap: &'a mut MmapMut, shape: Vec<usize>) -> Result<Self> {
        let expected_bytes = shape.iter().product::<usize>() * mem::size_of::<T>();
        if mmap.len() < expected_bytes {
            return Err(IoError::FormatError(format!(
                "Memory map too small: expected {} bytes, got {}",
                expected_bytes,
                mmap.len()
            )));
        }
        
        Ok(Self {
            mmap,
            shape,
            _phantom: PhantomData,
        })
    }
    
    /// Get a mutable ndarray view without copying data
    pub fn as_array_view_mut(&mut self) -> ArrayViewMut<T, IxDyn> {
        let ptr = self.mmap.as_mut_ptr() as *mut T;
        let slice = unsafe {
            slice::from_raw_parts_mut(ptr, self.shape.iter().product())
        };
        
        ArrayViewMut::from_shape(IxDyn(&self.shape), slice)
            .expect("Shape mismatch in zero-copy view")
    }
    
    /// Get a mutable slice view
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let ptr = self.mmap.as_mut_ptr() as *mut T;
        let len = self.shape.iter().product();
        unsafe { slice::from_raw_parts_mut(ptr, len) }
    }
}

/// Zero-copy file reader using memory mapping
pub struct ZeroCopyReader {
    file: File,
    mmap: Option<Mmap>,
}

impl ZeroCopyReader {
    /// Create a new zero-copy reader
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
        Ok(Self { file, mmap: None })
    }
    
    /// Memory-map the entire file
    pub fn map_file(&mut self) -> Result<&Mmap> {
        if self.mmap.is_none() {
            let mmap = unsafe { MmapOptions::new().map(&self.file).map_err(|e| IoError::FileError(e.to_string()))? };
            self.mmap = Some(mmap);
        }
        Ok(self.mmap.as_ref().unwrap())
    }
    
    /// Read a zero-copy array view
    pub fn read_array<T>(&mut self, shape: Vec<usize>) -> Result<ZeroCopyArrayView<T>>
    where
        T: 'static + Copy,
    {
        let mmap = self.map_file()?;
        ZeroCopyArrayView::new(mmap, shape)
    }
    
    /// Read a portion of the file without copying
    pub fn read_slice(&mut self, offset: usize, len: usize) -> Result<&[u8]> {
        let mmap = self.map_file()?;
        if offset + len > mmap.len() {
            return Err(IoError::Other(
                "Slice extends beyond file boundaries".to_string(),
            ));
        }
        Ok(&mmap[offset..offset + len])
    }
}

/// Zero-copy file writer using memory mapping
pub struct ZeroCopyWriter {
    file: File,
    mmap: Option<MmapMut>,
}

impl ZeroCopyWriter {
    /// Create a new zero-copy writer
    pub fn new<P: AsRef<Path>>(path: P, size: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path).map_err(|e| IoError::FileError(e.to_string()))?;
        
        // Set file size
        file.set_len(size as u64).map_err(|e| IoError::FileError(e.to_string()))?;
        
        Ok(Self { file, mmap: None })
    }
    
    /// Memory-map the file for writing
    pub fn map_file_mut(&mut self) -> Result<&mut MmapMut> {
        if self.mmap.is_none() {
            let mmap = unsafe { MmapOptions::new().map_mut(&self.file).map_err(|e| IoError::FileError(e.to_string()))? };
            self.mmap = Some(mmap);
        }
        Ok(self.mmap.as_mut().unwrap())
    }
    
    /// Write an array without copying
    pub fn write_array<T>(&mut self, shape: Vec<usize>) -> Result<ZeroCopyArrayViewMut<T>>
    where
        T: 'static + Copy,
    {
        let mmap = self.map_file_mut()?;
        ZeroCopyArrayViewMut::new(mmap, shape)
    }
    
    /// Write to a slice without copying
    pub fn write_slice(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let mmap = self.map_file_mut()?;
        if offset + data.len() > mmap.len() {
            return Err(IoError::Other(
                "Write extends beyond file boundaries".to_string(),
            ));
        }
        mmap[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }
    
    /// Flush changes to disk
    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mut mmap) = self.mmap {
            mmap.flush().map_err(|e| IoError::FileError(e.to_string()))?;
        }
        Ok(())
    }
}

/// Zero-copy CSV reader
pub struct ZeroCopyCsvReader<'a> {
    data: &'a [u8],
    delimiter: u8,
}

impl<'a> ZeroCopyCsvReader<'a> {
    /// Create a new zero-copy CSV reader
    pub fn new(data: &'a [u8], delimiter: u8) -> Self {
        Self { data, delimiter }
    }
    
    /// Iterate over lines without allocating
    pub fn lines(&self) -> ZeroCopyLineIterator<'a> {
        ZeroCopyLineIterator {
            data: self.data,
            pos: 0,
        }
    }
    
    /// Parse a line into fields without allocating
    pub fn parse_line(&self, line: &'a [u8]) -> Vec<&'a str> {
        let mut fields = Vec::new();
        let mut start = 0;
        
        for (i, &byte) in line.iter().enumerate() {
            if byte == self.delimiter {
                if let Ok(field) = std::str::from_utf8(&line[start..i]) {
                    fields.push(field);
                }
                start = i + 1;
            }
        }
        
        // Add last field
        if start < line.len() {
            if let Ok(field) = std::str::from_utf8(&line[start..]) {
                fields.push(field);
            }
        }
        
        fields
    }
}

/// Iterator over lines without allocation
pub struct ZeroCopyLineIterator<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for ZeroCopyLineIterator<'a> {
    type Item = &'a [u8];
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        
        let line = &self.data[start..self.pos];
        
        // Skip newline
        if self.pos < self.data.len() {
            self.pos += 1;
        }
        
        Some(line)
    }
}

/// Zero-copy binary format reader
pub struct ZeroCopyBinaryReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ZeroCopyBinaryReader<'a> {
    /// Create a new zero-copy binary reader
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    
    /// Read a value without copying
    pub fn read<T: Copy>(&mut self) -> Result<T> {
        let size = mem::size_of::<T>();
        if self.pos + size > self.data.len() {
            return Err(IoError::Other("Not enough data".to_string()));
        }
        
        let value = unsafe {
            let ptr = self.data.as_ptr().add(self.pos) as *const T;
            ptr.read_unaligned()
        };
        
        self.pos += size;
        Ok(value)
    }
    
    /// Read a slice without copying
    pub fn read_slice(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.pos + len > self.data.len() {
            return Err(IoError::Other("Not enough data".to_string()));
        }
        
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }
    
    /// Get remaining data
    pub fn remaining(&self) -> &'a [u8] {
        &self.data[self.pos..]
    }
    
    /// Read an array of f32 values using SIMD optimization
    pub fn read_f32_array_simd(&mut self, count: usize) -> Result<Array1<f32>> {
        let bytes_needed = count * mem::size_of::<f32>();
        if self.pos + bytes_needed > self.data.len() {
            return Err(IoError::Other("Not enough data for f32 array".to_string()));
        }
        
        let slice = unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.pos) as *const f32,
                count
            )
        };
        
        self.pos += bytes_needed;
        Ok(Array1::from_vec(slice.to_vec()))
    }
    
    /// Read an array of f64 values using SIMD optimization
    pub fn read_f64_array_simd(&mut self, count: usize) -> Result<Array1<f64>> {
        let bytes_needed = count * mem::size_of::<f64>();
        if self.pos + bytes_needed > self.data.len() {
            return Err(IoError::Other("Not enough data for f64 array".to_string()));
        }
        
        let slice = unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.pos) as *const f64,
                count
            )
        };
        
        self.pos += bytes_needed;
        Ok(Array1::from_vec(slice.to_vec()))
    }
}

/// SIMD-optimized zero-copy operations
pub mod simd_zero_copy {
    use super::*;
    use ndarray::{Array2, ArrayView2};
    
    /// Zero-copy SIMD operations for f32 arrays
    pub struct SimdZeroCopyOpsF32;
    
    impl SimdZeroCopyOpsF32 {
        /// Perform element-wise addition on memory-mapped arrays
        pub fn add_mmap(a_mmap: &Mmap, b_mmap: &Mmap, shape: &[usize]) -> Result<Array1<f32>> {
            if a_mmap.len() != b_mmap.len() {
                return Err(IoError::Other("Memory maps must have same size".to_string()));
            }
            
            let count = shape.iter().product::<usize>();
            let expected_bytes = count * mem::size_of::<f32>();
            
            if a_mmap.len() < expected_bytes {
                return Err(IoError::Other("Memory map too small for shape".to_string()));
            }
            
            // Create array views from memory maps
            let a_slice = unsafe {
                slice::from_raw_parts(a_mmap.as_ptr() as *const f32, count)
            };
            let b_slice = unsafe {
                slice::from_raw_parts(b_mmap.as_ptr() as *const f32, count)
            };
            
            let a_view = ArrayView1::from_shape(count, a_slice).unwrap();
            let b_view = ArrayView1::from_shape(count, b_slice).unwrap();
            
            // Use SIMD operations
            Ok(f32::simd_add(&a_view, &b_view))
        }
        
        /// Perform scalar multiplication on a memory-mapped array
        pub fn scalar_mul_mmap(mmap: &Mmap, scalar: f32, shape: &[usize]) -> Result<Array1<f32>> {
            let count = shape.iter().product::<usize>();
            let expected_bytes = count * mem::size_of::<f32>();
            
            if mmap.len() < expected_bytes {
                return Err(IoError::Other("Memory map too small for shape".to_string()));
            }
            
            let slice = unsafe {
                slice::from_raw_parts(mmap.as_ptr() as *const f32, count)
            };
            
            let view = ArrayView1::from_shape(count, slice).unwrap();
            
            // Use SIMD scalar multiplication
            Ok(f32::simd_scalar_mul(&view, scalar))
        }
        
        /// Compute dot product directly from memory-mapped arrays
        pub fn dot_mmap(a_mmap: &Mmap, b_mmap: &Mmap, len: usize) -> Result<f32> {
            let expected_bytes = len * mem::size_of::<f32>();
            
            if a_mmap.len() < expected_bytes || b_mmap.len() < expected_bytes {
                return Err(IoError::Other("Memory maps too small".to_string()));
            }
            
            let a_slice = unsafe {
                slice::from_raw_parts(a_mmap.as_ptr() as *const f32, len)
            };
            let b_slice = unsafe {
                slice::from_raw_parts(b_mmap.as_ptr() as *const f32, len)
            };
            
            let a_view = ArrayView1::from_shape(len, a_slice).unwrap();
            let b_view = ArrayView1::from_shape(len, b_slice).unwrap();
            
            // Use SIMD dot product
            Ok(f32::simd_dot(&a_view, &b_view))
        }
    }
    
    /// Zero-copy SIMD operations for f64 arrays
    pub struct SimdZeroCopyOpsF64;
    
    impl SimdZeroCopyOpsF64 {
        /// Perform element-wise addition on memory-mapped arrays
        pub fn add_mmap(a_mmap: &Mmap, b_mmap: &Mmap, shape: &[usize]) -> Result<Array1<f64>> {
            if a_mmap.len() != b_mmap.len() {
                return Err(IoError::Other("Memory maps must have same size".to_string()));
            }
            
            let count = shape.iter().product::<usize>();
            let expected_bytes = count * mem::size_of::<f64>();
            
            if a_mmap.len() < expected_bytes {
                return Err(IoError::Other("Memory map too small for shape".to_string()));
            }
            
            // Create array views from memory maps
            let a_slice = unsafe {
                slice::from_raw_parts(a_mmap.as_ptr() as *const f64, count)
            };
            let b_slice = unsafe {
                slice::from_raw_parts(b_mmap.as_ptr() as *const f64, count)
            };
            
            let a_view = ArrayView1::from_shape(count, a_slice).unwrap();
            let b_view = ArrayView1::from_shape(count, b_slice).unwrap();
            
            // Use SIMD operations
            Ok(f64::simd_add(&a_view, &b_view))
        }
        
        /// Matrix multiplication directly from memory-mapped files
        pub fn gemm_mmap(
            a_mmap: &Mmap,
            b_mmap: &Mmap,
            a_shape: (usize, usize),
            b_shape: (usize, usize),
            alpha: f64,
            beta: f64,
        ) -> Result<Array2<f64>> {
            let (m, k1) = a_shape;
            let (k2, n) = b_shape;
            
            if k1 != k2 {
                return Err(IoError::Other("Matrix dimensions don't match for multiplication".to_string()));
            }
            
            let a_expected = m * k1 * mem::size_of::<f64>();
            let b_expected = k2 * n * mem::size_of::<f64>();
            
            if a_mmap.len() < a_expected || b_mmap.len() < b_expected {
                return Err(IoError::Other("Memory maps too small for matrices".to_string()));
            }
            
            // Create array views
            let a_slice = unsafe {
                slice::from_raw_parts(a_mmap.as_ptr() as *const f64, m * k1)
            };
            let b_slice = unsafe {
                slice::from_raw_parts(b_mmap.as_ptr() as *const f64, k2 * n)
            };
            
            let a_view = ArrayView2::from_shape((m, k1), a_slice).unwrap();
            let b_view = ArrayView2::from_shape((k2, n), b_slice).unwrap();
            
            let mut c = Array2::<f64>::zeros((m, n));
            
            // Use SIMD GEMM
            f64::simd_gemm(alpha, &a_view, &b_view, beta, &mut c);
            
            Ok(c)
        }
    }
}

/// Zero-copy streaming processor for large datasets
pub struct ZeroCopyStreamProcessor<T> {
    reader: ZeroCopyReader,
    chunk_size: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy + 'static> ZeroCopyStreamProcessor<T> {
    /// Create a new streaming processor
    pub fn new<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        let reader = ZeroCopyReader::new(path)?;
        Ok(Self {
            reader,
            chunk_size,
            _phantom: PhantomData,
        })
    }
    
    /// Process the file in chunks using parallel processing
    pub fn process_parallel<F, R>(&mut self, shape: Vec<usize>, processor: F) -> Result<Vec<R>>
    where
        F: Fn(&[T]) -> R + Send + Sync,
        R: Send,
        T: Send + Sync,
    {
        let capabilities = PlatformCapabilities::detect();
        let mmap = self.reader.map_file()?;
        
        let total_elements: usize = shape.iter().product();
        let element_size = mem::size_of::<T>();
        let total_bytes = total_elements * element_size;
        
        if mmap.len() < total_bytes {
            return Err(IoError::Other("File too small for specified shape".to_string()));
        }
        
        // Create chunks for parallel processing
        let ptr = mmap.as_ptr() as *const T;
        let data_slice = unsafe { slice::from_raw_parts(ptr, total_elements) };
        
        if capabilities.simd_available && total_elements > 10000 {
            // Use parallel processing for large datasets
            let results: Vec<R> = data_slice
                .chunks(self.chunk_size)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|chunk| processor(chunk))
                .collect();
            
            Ok(results)
        } else {
            // Sequential processing for smaller datasets
            let results: Vec<R> = data_slice
                .chunks(self.chunk_size)
                .map(|chunk| processor(chunk))
                .collect();
            
            Ok(results)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_zero_copy_reader() -> Result<()> {
        // Create a temporary file with data
        let mut file = NamedTempFile::new().map_err(|e| IoError::FileError(e.to_string()))?;
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let bytes = unsafe {
            slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
        };
        file.write_all(bytes).map_err(|e| IoError::FileError(e.to_string()))?;
        
        // Read using zero-copy
        let mut reader = ZeroCopyReader::new(file.path())?;
        let array_view = reader.read_array::<f64>(vec![10, 10])?;
        let view = array_view.as_array_view();
        
        assert_eq!(view.shape(), &[10, 10]);
        assert_eq!(view[[0, 0]], 0.0);
        assert_eq!(view[[9, 9]], 99.0);
        
        Ok(())
    }
    
    #[test]
    fn test_zero_copy_csv() {
        let data = b"a,b,c\n1,2,3\n4,5,6";
        let reader = ZeroCopyCsvReader::new(data, b',');
        
        let lines: Vec<_> = reader.lines().collect();
        assert_eq!(lines.len(), 3);
        
        let fields = reader.parse_line(lines[0]);
        assert_eq!(fields, vec!["a", "b", "c"]);
    }
    
    #[test]
    fn test_simd_zero_copy_add() -> Result<()> {
        // Create two temporary files with f32 data
        let mut file1 = NamedTempFile::new().map_err(|e| IoError::FileError(e.to_string()))?;
        let mut file2 = NamedTempFile::new().map_err(|e| IoError::FileError(e.to_string()))?;
        
        let data1: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let data2: Vec<f32> = (0..100).map(|i| (i * 2) as f32).collect();
        
        let bytes1 = unsafe {
            slice::from_raw_parts(data1.as_ptr() as *const u8, data1.len() * 4)
        };
        let bytes2 = unsafe {
            slice::from_raw_parts(data2.as_ptr() as *const u8, data2.len() * 4)
        };
        
        file1.write_all(bytes1).map_err(|e| IoError::FileError(e.to_string()))?;
        file2.write_all(bytes2).map_err(|e| IoError::FileError(e.to_string()))?;
        
        // Memory map both files
        let mmap1 = unsafe { MmapOptions::new().map(&file1).map_err(|e| IoError::FileError(e.to_string()))? };
        let mmap2 = unsafe { MmapOptions::new().map(&file2).map_err(|e| IoError::FileError(e.to_string()))? };
        
        // Perform SIMD addition
        let result = simd_zero_copy::SimdZeroCopyOpsF32::add_mmap(&mmap1, &mmap2, &[100])?;
        
        // Verify results
        assert_eq!(result.len(), 100);
        assert_eq!(result[0], 0.0);  // 0 + 0
        assert_eq!(result[50], 150.0); // 50 + 100
        assert_eq!(result[99], 297.0); // 99 + 198
        
        Ok(())
    }
}