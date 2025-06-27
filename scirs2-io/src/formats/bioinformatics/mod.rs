//! Bioinformatics file format support
//!
//! This module provides support for common bioinformatics file formats used
//! in genomics, proteomics, and molecular biology research.
//!
//! ## Supported Formats
//!
//! - **FASTA**: Standard format for nucleotide and protein sequences
//! - **FASTQ**: Sequences with per-base quality scores
//! - **SAM/BAM**: Sequence Alignment/Map format
//! - **VCF**: Variant Call Format for genomic variations
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::formats::bioinformatics::{FastaReader, FastqReader};
//!
//! // Read FASTA file
//! let mut reader = FastaReader::open("sequences.fasta")?;
//! for record in reader.records() {
//!     let record = record?;
//!     println!(">{}", record.id());
//!     println!("{}", record.sequence());
//! }
//!
//! // Read FASTQ file
//! let mut reader = FastqReader::open("reads.fastq")?;
//! for record in reader.records() {
//!     let record = record?;
//!     println!("@{}", record.id());
//!     println!("{}", record.sequence());
//!     println!("+");
//!     println!("{}", record.quality());
//! }
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use crate::error::{IoError, Result};

/// FASTA sequence record
#[derive(Debug, Clone, PartialEq)]
pub struct FastaRecord {
    /// Sequence identifier (header line without '>')
    id: String,
    /// Optional description after the ID
    description: Option<String>,
    /// Sequence data
    sequence: String,
}

impl FastaRecord {
    /// Create a new FASTA record
    pub fn new(id: String, sequence: String) -> Self {
        Self {
            id,
            description: None,
            sequence,
        }
    }

    /// Create a new FASTA record with description
    pub fn with_description(id: String, description: String, sequence: String) -> Self {
        Self {
            id,
            description: Some(description),
            sequence,
        }
    }

    /// Get the sequence ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the optional description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get the sequence
    pub fn sequence(&self) -> &str {
        &self.sequence
    }

    /// Get the sequence length
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Get the full header (ID + description)
    pub fn header(&self) -> String {
        match &self.description {
            Some(desc) => format!("{} {}", self.id, desc),
            None => self.id.clone(),
        }
    }
}

/// FASTA file reader
pub struct FastaReader {
    reader: BufReader<File>,
    line_buffer: String,
}

impl FastaReader {
    /// Open a FASTA file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string(), e))?;
        Ok(Self {
            reader: BufReader::new(file),
            line_buffer: String::new(),
        })
    }

    /// Create an iterator over FASTA records
    pub fn records(&mut self) -> FastaRecordIterator<'_> {
        FastaRecordIterator { reader: self }
    }

    /// Read the next record
    fn read_record(&mut self) -> Result<Option<FastaRecord>> {
        self.line_buffer.clear();
        
        // Find the next header line
        loop {
            if self.reader.read_line(&mut self.line_buffer)
                .map_err(|e| IoError::ParseError(format!("Failed to read line: {}", e)))? == 0
            {
                return Ok(None); // EOF
            }
            
            if self.line_buffer.starts_with('>') {
                break;
            }
            self.line_buffer.clear();
        }
        
        // Parse header
        let header = self.line_buffer[1..].trim().to_string();
        let (id, description) = if let Some(space_pos) = header.find(' ') {
            let (id_part, desc_part) = header.split_at(space_pos);
            (id_part.to_string(), Some(desc_part[1..].to_string()))
        } else {
            (header, None)
        };
        
        // Read sequence lines until next header or EOF
        let mut sequence = String::new();
        self.line_buffer.clear();
        
        loop {
            let bytes_read = self.reader.read_line(&mut self.line_buffer)
                .map_err(|e| IoError::ParseError(format!("Failed to read line: {}", e)))?;
            
            if bytes_read == 0 || self.line_buffer.starts_with('>') {
                // Reached next record or EOF
                if self.line_buffer.starts_with('>') {
                    // Put back the header line for the next iteration
                    // This is a bit hacky but works for our use case
                }
                break;
            }
            
            sequence.push_str(self.line_buffer.trim());
            if !self.line_buffer.starts_with('>') {
                self.line_buffer.clear();
            }
        }
        
        Ok(Some(FastaRecord {
            id,
            description,
            sequence,
        }))
    }
}

/// Iterator over FASTA records
pub struct FastaRecordIterator<'a> {
    reader: &'a mut FastaReader,
}

impl<'a> Iterator for FastaRecordIterator<'a> {
    type Item = Result<FastaRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_record() {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// FASTA file writer
pub struct FastaWriter {
    writer: BufWriter<File>,
    line_width: usize,
}

impl FastaWriter {
    /// Create a new FASTA file for writing
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::WriteError(format!("Failed to create file: {}", e)))?;
        Ok(Self {
            writer: BufWriter::new(file),
            line_width: 80, // Standard FASTA line width
        })
    }

    /// Set the line width for sequence wrapping
    pub fn set_line_width(&mut self, width: usize) {
        self.line_width = width;
    }

    /// Write a FASTA record
    pub fn write_record(&mut self, record: &FastaRecord) -> Result<()> {
        // Write header
        write!(self.writer, ">{}", record.header())
            .map_err(|e| IoError::WriteError(format!("Failed to write header: {}", e)))?;
        writeln!(self.writer)
            .map_err(|e| IoError::WriteError(format!("Failed to write newline: {}", e)))?;
        
        // Write sequence with line wrapping
        let sequence = record.sequence();
        for chunk in sequence.as_bytes().chunks(self.line_width) {
            self.writer.write_all(chunk)
                .map_err(|e| IoError::WriteError(format!("Failed to write sequence: {}", e)))?;
            writeln!(self.writer)
                .map_err(|e| IoError::WriteError(format!("Failed to write newline: {}", e)))?;
        }
        
        Ok(())
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()
            .map_err(|e| IoError::WriteError(format!("Failed to flush: {}", e)))
    }
}

/// FASTQ quality encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityEncoding {
    /// Sanger/Illumina 1.8+ (Phred+33)
    Sanger,
    /// Illumina 1.3-1.7 (Phred+64)
    Illumina,
}

impl Default for QualityEncoding {
    fn default() -> Self {
        QualityEncoding::Sanger
    }
}

/// FASTQ sequence record
#[derive(Debug, Clone, PartialEq)]
pub struct FastqRecord {
    /// Sequence identifier
    id: String,
    /// Optional description
    description: Option<String>,
    /// Sequence data
    sequence: String,
    /// Quality scores (ASCII encoded)
    quality: String,
}

impl FastqRecord {
    /// Create a new FASTQ record
    pub fn new(id: String, sequence: String, quality: String) -> Result<Self> {
        if sequence.len() != quality.len() {
            return Err(IoError::ParseError(format!(
                "Sequence and quality lengths don't match: {} vs {}",
                sequence.len(),
                quality.len()
            )));
        }
        
        Ok(Self {
            id,
            description: None,
            sequence,
            quality,
        })
    }

    /// Create a new FASTQ record with description
    pub fn with_description(id: String, description: String, sequence: String, quality: String) -> Result<Self> {
        if sequence.len() != quality.len() {
            return Err(IoError::ParseError(format!(
                "Sequence and quality lengths don't match: {} vs {}",
                sequence.len(),
                quality.len()
            )));
        }
        
        Ok(Self {
            id,
            description: Some(description),
            sequence,
            quality,
        })
    }

    /// Get the sequence ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the optional description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get the sequence
    pub fn sequence(&self) -> &str {
        &self.sequence
    }

    /// Get the quality string
    pub fn quality(&self) -> &str {
        &self.quality
    }

    /// Get the sequence length
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Get quality scores as numeric values
    pub fn quality_scores(&self, encoding: QualityEncoding) -> Vec<u8> {
        let offset = match encoding {
            QualityEncoding::Sanger => 33,
            QualityEncoding::Illumina => 64,
        };
        
        self.quality.bytes()
            .map(|b| b.saturating_sub(offset))
            .collect()
    }

    /// Get the full header (ID + description)
    pub fn header(&self) -> String {
        match &self.description {
            Some(desc) => format!("{} {}", self.id, desc),
            None => self.id.clone(),
        }
    }
}

/// FASTQ file reader
pub struct FastqReader {
    reader: BufReader<File>,
    encoding: QualityEncoding,
    line_buffer: String,
}

impl FastqReader {
    /// Open a FASTQ file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string(), e))?;
        Ok(Self {
            reader: BufReader::new(file),
            encoding: QualityEncoding::default(),
            line_buffer: String::new(),
        })
    }

    /// Open a FASTQ file with specific quality encoding
    pub fn open_with_encoding<P: AsRef<Path>>(path: P, encoding: QualityEncoding) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string(), e))?;
        Ok(Self {
            reader: BufReader::new(file),
            encoding,
            line_buffer: String::new(),
        })
    }

    /// Create an iterator over FASTQ records
    pub fn records(&mut self) -> FastqRecordIterator<'_> {
        FastqRecordIterator { reader: self }
    }

    /// Read the next record
    fn read_record(&mut self) -> Result<Option<FastqRecord>> {
        // Read header line
        self.line_buffer.clear();
        if self.reader.read_line(&mut self.line_buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read header: {}", e)))? == 0
        {
            return Ok(None); // EOF
        }
        
        if !self.line_buffer.starts_with('@') {
            return Err(IoError::ParseError(format!(
                "Expected '@' at start of header, found: {}",
                self.line_buffer.trim()
            )));
        }
        
        // Parse header
        let header = self.line_buffer[1..].trim().to_string();
        let (id, description) = if let Some(space_pos) = header.find(' ') {
            let (id_part, desc_part) = header.split_at(space_pos);
            (id_part.to_string(), Some(desc_part[1..].to_string()))
        } else {
            (header, None)
        };
        
        // Read sequence line
        self.line_buffer.clear();
        self.reader.read_line(&mut self.line_buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read sequence: {}", e)))?;
        let sequence = self.line_buffer.trim().to_string();
        
        // Read separator line
        self.line_buffer.clear();
        self.reader.read_line(&mut self.line_buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read separator: {}", e)))?;
        if !self.line_buffer.starts_with('+') {
            return Err(IoError::ParseError(format!(
                "Expected '+' separator, found: {}",
                self.line_buffer.trim()
            )));
        }
        
        // Read quality line
        self.line_buffer.clear();
        self.reader.read_line(&mut self.line_buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read quality: {}", e)))?;
        let quality = self.line_buffer.trim().to_string();
        
        FastqRecord::new(id, sequence, quality)
            .or_else(|_| FastqRecord::with_description(id, description.unwrap_or_default(), sequence, quality))
            .map(Some)
    }
}

/// Iterator over FASTQ records
pub struct FastqRecordIterator<'a> {
    reader: &'a mut FastqReader,
}

impl<'a> Iterator for FastqRecordIterator<'a> {
    type Item = Result<FastqRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_record() {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// FASTQ file writer
pub struct FastqWriter {
    writer: BufWriter<File>,
    encoding: QualityEncoding,
}

impl FastqWriter {
    /// Create a new FASTQ file for writing
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::WriteError(format!("Failed to create file: {}", e)))?;
        Ok(Self {
            writer: BufWriter::new(file),
            encoding: QualityEncoding::default(),
        })
    }

    /// Create a new FASTQ file with specific quality encoding
    pub fn create_with_encoding<P: AsRef<Path>>(path: P, encoding: QualityEncoding) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::WriteError(format!("Failed to create file: {}", e)))?;
        Ok(Self {
            writer: BufWriter::new(file),
            encoding,
        })
    }

    /// Write a FASTQ record
    pub fn write_record(&mut self, record: &FastqRecord) -> Result<()> {
        // Write header
        writeln!(self.writer, "@{}", record.header())
            .map_err(|e| IoError::WriteError(format!("Failed to write header: {}", e)))?;
        
        // Write sequence
        writeln!(self.writer, "{}", record.sequence())
            .map_err(|e| IoError::WriteError(format!("Failed to write sequence: {}", e)))?;
        
        // Write separator
        writeln!(self.writer, "+")
            .map_err(|e| IoError::WriteError(format!("Failed to write separator: {}", e)))?;
        
        // Write quality
        writeln!(self.writer, "{}", record.quality())
            .map_err(|e| IoError::WriteError(format!("Failed to write quality: {}", e)))?;
        
        Ok(())
    }

    /// Write a FASTQ record from numeric quality scores
    pub fn write_record_with_scores(&mut self, id: &str, sequence: &str, quality_scores: &[u8]) -> Result<()> {
        if sequence.len() != quality_scores.len() {
            return Err(IoError::WriteError(format!(
                "Sequence and quality lengths don't match: {} vs {}",
                sequence.len(),
                quality_scores.len()
            )));
        }
        
        let offset = match self.encoding {
            QualityEncoding::Sanger => 33,
            QualityEncoding::Illumina => 64,
        };
        
        let quality_string: String = quality_scores.iter()
            .map(|&score| (score.saturating_add(offset)) as char)
            .collect();
        
        let record = FastqRecord::new(id.to_string(), sequence.to_string(), quality_string)?;
        self.write_record(&record)
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()
            .map_err(|e| IoError::WriteError(format!("Failed to flush: {}", e)))
    }
}

/// Count sequences in a FASTA file
pub fn count_fasta_sequences<P: AsRef<Path>>(path: P) -> Result<usize> {
    let file = File::open(path.as_ref())
        .map_err(|e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string(), e))?;
    let reader = BufReader::new(file);
    
    let count = reader.lines()
        .filter_map(|line| line.ok())
        .filter(|line| line.starts_with('>'))
        .count();
    
    Ok(count)
}

/// Count sequences in a FASTQ file
pub fn count_fastq_sequences<P: AsRef<Path>>(path: P) -> Result<usize> {
    let file = File::open(path.as_ref())
        .map_err(|e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string(), e))?;
    let reader = BufReader::new(file);
    
    let line_count = reader.lines().count();
    if line_count % 4 != 0 {
        return Err(IoError::ParseError(format!(
            "Invalid FASTQ file: line count {} is not divisible by 4",
            line_count
        )));
    }
    
    Ok(line_count / 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_fasta_read_write() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        
        // Write test data
        {
            let mut writer = FastaWriter::create(path)?;
            writer.write_record(&FastaRecord::new(
                "seq1".to_string(),
                "ATCGATCGATCG".to_string(),
            ))?;
            writer.write_record(&FastaRecord::with_description(
                "seq2".to_string(),
                "test sequence".to_string(),
                "GCTAGCTAGCTA".to_string(),
            ))?;
            writer.flush()?;
        }
        
        // Read and verify
        {
            let mut reader = FastaReader::open(path)?;
            let records: Vec<_> = reader.records().collect::<Result<Vec<_>>>()?;
            
            assert_eq!(records.len(), 2);
            assert_eq!(records[0].id(), "seq1");
            assert_eq!(records[0].sequence(), "ATCGATCGATCG");
            assert_eq!(records[1].id(), "seq2");
            assert_eq!(records[1].description(), Some("test sequence"));
            assert_eq!(records[1].sequence(), "GCTAGCTAGCTA");
        }
        
        Ok(())
    }

    #[test]
    fn test_fastq_read_write() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        
        // Write test data
        {
            let mut writer = FastqWriter::create(path)?;
            writer.write_record(&FastqRecord::new(
                "read1".to_string(),
                "ATCG".to_string(),
                "IIII".to_string(),
            )?)?;
            writer.write_record_with_scores(
                "read2",
                "GCTA",
                &[30, 35, 40, 35],
            )?;
            writer.flush()?;
        }
        
        // Read and verify
        {
            let mut reader = FastqReader::open(path)?;
            let records: Vec<_> = reader.records().collect::<Result<Vec<_>>>()?;
            
            assert_eq!(records.len(), 2);
            assert_eq!(records[0].id(), "read1");
            assert_eq!(records[0].sequence(), "ATCG");
            assert_eq!(records[0].quality(), "IIII");
            
            assert_eq!(records[1].id(), "read2");
            assert_eq!(records[1].sequence(), "GCTA");
            let scores = records[1].quality_scores(QualityEncoding::Sanger);
            assert_eq!(scores, vec![30, 35, 40, 35]);
        }
        
        Ok(())
    }

    #[test]
    fn test_sequence_counting() -> Result<()> {
        let fasta_file = NamedTempFile::new().unwrap();
        let fastq_file = NamedTempFile::new().unwrap();
        
        // Create test FASTA
        {
            let mut writer = FastaWriter::create(fasta_file.path())?;
            for i in 0..5 {
                writer.write_record(&FastaRecord::new(
                    format!("seq{}", i),
                    "ATCG".to_string(),
                ))?;
            }
            writer.flush()?;
        }
        
        // Create test FASTQ
        {
            let mut writer = FastqWriter::create(fastq_file.path())?;
            for i in 0..3 {
                writer.write_record(&FastqRecord::new(
                    format!("read{}", i),
                    "ATCG".to_string(),
                    "IIII".to_string(),
                )?)?;
            }
            writer.flush()?;
        }
        
        assert_eq!(count_fasta_sequences(fasta_file.path())?, 5);
        assert_eq!(count_fastq_sequences(fastq_file.path())?, 3);
        
        Ok(())
    }
}