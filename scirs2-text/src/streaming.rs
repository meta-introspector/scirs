//! Memory-efficient streaming and memory-mapped text processing
//!
//! This module provides utilities for processing large text corpora that don't fit in memory
//! using streaming and memory-mapped file techniques.

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vocabulary::Vocabulary;
use crate::sparse::{SparseVector, SparseMatrixBuilder, CsrMatrix};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use memmap2::{Mmap, MmapOptions};

/// Memory-mapped corpus for efficient large file processing
pub struct MemoryMappedCorpus {
    mmap: Arc<Mmap>,
    line_offsets: Vec<usize>,
}

impl MemoryMappedCorpus {
    /// Create a new memory-mapped corpus from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| 
            TextError::IoError(format!("Failed to open file: {}", e))
        )?;
        
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| TextError::IoError(format!("Failed to memory map file: {}", e)))?
        };
        
        // Build line offset index
        let line_offsets = Self::build_line_index(&mmap);
        
        Ok(Self {
            mmap: Arc::new(mmap),
            line_offsets,
        })
    }
    
    /// Build an index of line offsets for fast access
    fn build_line_index(mmap: &Mmap) -> Vec<usize> {
        let mut offsets = vec![0];
        let data = mmap.as_ref();
        
        for (i, &byte) in data.iter().enumerate() {
            if byte == b'\n' {
                offsets.push(i + 1);
            }
        }
        
        offsets
    }
    
    /// Get the number of documents (lines) in the corpus
    pub fn num_documents(&self) -> usize {
        self.line_offsets.len().saturating_sub(1)
    }
    
    /// Get a specific document by index
    pub fn get_document(&self, index: usize) -> Result<&str> {
        if index >= self.num_documents() {
            return Err(TextError::InvalidInput(
                format!("Document index {} out of range", index)
            ));
        }
        
        let start = self.line_offsets[index];
        let end = if index + 1 < self.line_offsets.len() {
            self.line_offsets[index + 1].saturating_sub(1) // Remove newline
        } else {
            self.mmap.len()
        };
        
        let data = &self.mmap[start..end];
        std::str::from_utf8(data).map_err(|e| 
            TextError::IoError(format!("Invalid UTF-8 in document: {}", e))
        )
    }
    
    /// Iterate over all documents
    pub fn iter(&self) -> CorpusIterator {
        CorpusIterator {
            corpus: self,
            current: 0,
        }
    }
    
    /// Process documents in parallel chunks
    pub fn parallel_process<F, R>(&self, chunk_size: usize, processor: F) -> Result<Vec<R>>
    where
        F: Fn(&[&str]) -> Result<R> + Send + Sync,
        R: Send,
    {
        use scirs2_core::parallel_ops::*;
        
        let num_docs = self.num_documents();
        let num_chunks = (num_docs + chunk_size - 1) / chunk_size;
        
        (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(num_docs);
                
                let mut docs = Vec::with_capacity(end - start);
                for i in start..end {
                    docs.push(self.get_document(i)?);
                }
                
                processor(&docs)
            })
            .collect()
    }
}

/// Iterator over documents in a memory-mapped corpus
pub struct CorpusIterator<'a> {
    corpus: &'a MemoryMappedCorpus,
    current: usize,
}

impl<'a> Iterator for CorpusIterator<'a> {
    type Item = Result<&'a str>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.corpus.num_documents() {
            return None;
        }
        
        let doc = self.corpus.get_document(self.current);
        self.current += 1;
        Some(doc)
    }
}

/// Streaming text processor for handling arbitrarily large files
pub struct StreamingTextProcessor<T: Tokenizer> {
    tokenizer: T,
    buffer_size: usize,
}

impl<T: Tokenizer> StreamingTextProcessor<T> {
    /// Create a new streaming processor
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            buffer_size: 1024 * 1024, // 1MB default buffer
        }
    }
    
    /// Set custom buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
    
    /// Process a file line by line
    pub fn process_lines<P, F, R>(&self, path: P, processor: F) -> Result<Vec<R>>
    where
        P: AsRef<Path>,
        F: FnMut(&str, usize) -> Result<Option<R>>,
    {
        let file = File::open(path).map_err(|e| 
            TextError::IoError(format!("Failed to open file: {}", e))
        )?;
        
        let reader = BufReader::with_capacity(self.buffer_size, file);
        self.process_reader_lines(reader, processor)
    }
    
    /// Process lines from any reader
    pub fn process_reader_lines<R: BufRead, F, U>(&self, reader: R, mut processor: F) -> Result<Vec<U>>
    where
        F: FnMut(&str, usize) -> Result<Option<U>>,
    {
        let mut results = Vec::new();
        let mut line_num = 0;
        
        for line_result in reader.lines() {
            let line = line_result.map_err(|e| 
                TextError::IoError(format!("Error reading line: {}", e))
            )?;
            
            if let Some(result) = processor(&line, line_num)? {
                results.push(result);
            }
            
            line_num += 1;
        }
        
        Ok(results)
    }
    
    /// Build vocabulary from a streaming corpus
    pub fn build_vocabulary_streaming<P: AsRef<Path>>(
        &self,
        path: P,
        min_count: usize,
    ) -> Result<Vocabulary> {
        let mut token_counts = HashMap::<String, usize>::new();
        
        // First pass: count tokens
        self.process_lines(&path, |line, _| {
            let tokens = self.tokenizer.tokenize(line)?;
            for token in tokens {
                *token_counts.entry(token).or_insert(0) += 1;
            }
            Ok(None::<()>)
        })?;
        
        // Build vocabulary with high-frequency tokens
        let mut vocab = Vocabulary::new();
        for (token, count) in &token_counts {
            if *count >= min_count {
                vocab.add_token(token);
            }
        }
        
        Ok(vocab)
    }
}

impl StreamingTextProcessor<WordTokenizer> {
    /// Create a streaming processor with default word tokenizer
    pub fn default() -> Self {
        Self::new(WordTokenizer::default())
    }
}

/// Streaming vectorizer for creating sparse matrices from large corpora
pub struct StreamingVectorizer {
    vocabulary: Vocabulary,
    chunk_size: usize,
}

impl StreamingVectorizer {
    /// Create a new streaming vectorizer
    pub fn new(vocabulary: Vocabulary) -> Self {
        Self {
            vocabulary,
            chunk_size: 1000, // Process 1000 documents at a time
        }
    }
    
    /// Set chunk size for processing
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }
    
    /// Transform a streaming corpus into a sparse matrix
    pub fn transform_streaming<P, T>(
        &self,
        path: P,
        tokenizer: &T,
    ) -> Result<CsrMatrix>
    where
        P: AsRef<Path>,
        T: Tokenizer,
    {
        let mut builder = SparseMatrixBuilder::new(self.vocabulary.len());
        
        let file = std::fs::File::open(path).map_err(|e| 
            TextError::IoError(format!("Failed to open file: {}", e))
        )?;
        let reader = std::io::BufReader::new(file);
        
        for line in reader.lines() {
            let line = line.map_err(|e| 
                TextError::IoError(format!("Error reading line: {}", e))
            )?;
            let tokens = tokenizer.tokenize(&line)?;
            let sparse_vec = self.tokens_to_sparse_vector(&tokens)?;
            builder.add_row(sparse_vec)?;
        }
        
        Ok(builder.build())
    }
    
    /// Convert tokens to sparse vector
    fn tokens_to_sparse_vector(&self, tokens: &[String]) -> Result<SparseVector> {
        let mut counts = std::collections::HashMap::new();
        
        for token in tokens {
            if let Some(idx) = self.vocabulary.get_index(token) {
                *counts.entry(idx).or_insert(0.0) += 1.0;
            }
        }
        
        let mut indices: Vec<usize> = counts.keys().copied().collect();
        indices.sort_unstable();
        
        let values: Vec<f64> = indices.iter().map(|&idx| counts[&idx]).collect();
        
        let sparse_vec = SparseVector::from_indices_values(indices, values, self.vocabulary.len());
        
        Ok(sparse_vec)
    }
}

/// Chunked corpus reader for processing files in manageable chunks
pub struct ChunkedCorpusReader {
    file: File,
    chunk_size: usize,
    position: u64,
    file_size: u64,
}

impl ChunkedCorpusReader {
    /// Create a new chunked reader
    pub fn new<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        let file = File::open(path).map_err(|e| 
            TextError::IoError(format!("Failed to open file: {}", e))
        )?;
        
        let file_size = file.metadata()
            .map_err(|e| TextError::IoError(format!("Failed to get file metadata: {}", e)))?
            .len();
        
        Ok(Self {
            file,
            chunk_size,
            position: 0,
            file_size,
        })
    }
    
    /// Read the next chunk of complete lines
    pub fn next_chunk(&mut self) -> Result<Option<Vec<String>>> {
        if self.position >= self.file_size {
            return Ok(None);
        }
        
        self.file.seek(SeekFrom::Start(self.position))
            .map_err(|e| TextError::IoError(format!("Failed to seek: {}", e)))?;
        
        let mut buffer = vec![0u8; self.chunk_size];
        let bytes_read = self.file.read(&mut buffer)
            .map_err(|e| TextError::IoError(format!("Failed to read chunk: {}", e)))?;
        
        if bytes_read == 0 {
            return Ok(None);
        }
        
        buffer.truncate(bytes_read);
        
        // Find the last newline to ensure complete lines
        let last_newline = buffer.iter().rposition(|&b| b == b'\n');
        
        let chunk_end = if let Some(pos) = last_newline {
            pos + 1
        } else if self.position + bytes_read as u64 >= self.file_size {
            bytes_read
        } else {
            // No newline found and not at end of file, need to read more
            return Err(TextError::IoError(
                "Chunk size too small to contain a complete line".into()
            ));
        };
        
        let chunk_str = std::str::from_utf8(&buffer[..chunk_end])
            .map_err(|e| TextError::IoError(format!("Invalid UTF-8: {}", e)))?;
        
        let lines: Vec<String> = chunk_str.lines().map(|s| s.to_string()).collect();
        
        self.position += chunk_end as u64;
        
        Ok(Some(lines))
    }
    
    /// Reset to the beginning of the file
    pub fn reset(&mut self) -> Result<()> {
        self.position = 0;
        self.file.seek(SeekFrom::Start(0))
            .map_err(|e| TextError::IoError(format!("Failed to seek: {}", e)))?;
        Ok(())
    }
}

/// Progress tracker for long-running operations
pub struct ProgressTracker {
    total: usize,
    current: usize,
    report_interval: usize,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(total: usize) -> Self {
        Self {
            total,
            current: 0,
            report_interval: total / 100, // Report every 1%
        }
    }
    
    /// Update progress
    pub fn update(&mut self, count: usize) {
        self.current += count;
        
        if self.current % self.report_interval == 0 || self.current >= self.total {
            let percentage = (self.current as f64 / self.total as f64) * 100.0;
            println!("Progress: {:.1}% ({}/{})", percentage, self.current, self.total);
        }
    }
    
    /// Check if complete
    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_memory_mapped_corpus() {
        // Create a temporary file with test data
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "First document").unwrap();
        writeln!(file, "Second document").unwrap();
        writeln!(file, "Third document").unwrap();
        file.flush().unwrap();
        
        let corpus = MemoryMappedCorpus::from_file(file.path()).unwrap();
        
        assert_eq!(corpus.num_documents(), 3);
        assert_eq!(corpus.get_document(0).unwrap(), "First document");
        assert_eq!(corpus.get_document(1).unwrap(), "Second document");
        assert_eq!(corpus.get_document(2).unwrap(), "Third document");
    }

    #[test]
    fn test_streaming_processor() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "hello world").unwrap();
        writeln!(file, "foo bar baz").unwrap();
        file.flush().unwrap();
        
        let processor = StreamingTextProcessor::default();
        let mut line_count = 0;
        
        processor.process_lines(file.path(), |_line, _num| {
            line_count += 1;
            Ok(None::<()>)
        }).unwrap();
        
        assert_eq!(line_count, 2);
    }

    #[test]
    fn test_chunked_reader() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..100 {
            writeln!(file, "Line {}", i).unwrap();
        }
        file.flush().unwrap();
        
        let mut reader = ChunkedCorpusReader::new(file.path(), 256).unwrap();
        let mut total_lines = 0;
        
        while let Some(lines) = reader.next_chunk().unwrap() {
            total_lines += lines.len();
        }
        
        assert_eq!(total_lines, 100);
    }

    #[test]
    fn test_streaming_vocabulary_building() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "the quick brown fox").unwrap();
        writeln!(file, "the lazy dog").unwrap();
        writeln!(file, "the brown dog").unwrap();
        file.flush().unwrap();
        
        let processor = StreamingTextProcessor::default();
        let vocab = processor.build_vocabulary_streaming(file.path(), 2).unwrap();
        
        // "the" appears 3 times, "brown" and "dog" appear 2 times each
        assert!(vocab.token_to_index("the").is_some());
        assert!(vocab.token_to_index("brown").is_some());
        assert!(vocab.token_to_index("dog").is_some());
        
        // These appear only once and should be pruned
        assert!(vocab.token_to_index("quick").is_none());
        assert!(vocab.token_to_index("fox").is_none());
        assert!(vocab.token_to_index("lazy").is_none());
    }
}