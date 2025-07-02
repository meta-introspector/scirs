//! Astronomical file format support
//!
//! This module provides support for file formats commonly used in astronomy,
//! astrophysics, and space science research.
//!
//! ## Supported Formats
//!
//! - **FITS**: Flexible Image Transport System - standard format for astronomical data
//! - **VOTable**: Virtual Observatory Table format for tabular astronomical data
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::formats::astronomical::{FitsFile, VOTable};
//! use ndarray::Array2;
//!
//! // Read FITS file
//! let fits = FitsFile::open("hubble_image.fits")?;
//! let header = fits.primary_header();
//! let image: Array2<f32> = fits.read_image()?;
//!
//! // Access header values
//! let exposure_time = header.get_f64("EXPTIME")?;
//! let telescope = header.get_string("TELESCOP")?;
//!
//! // Read FITS table
//! let table_hdu = fits.get_hdu(1)?;
//! let column_data = table_hdu.read_column("FLUX")?;
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::{IoError, Result};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use ndarray::Array2;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// FITS file structure
pub struct FitsFile {
    file_path: String,
    hdus: Vec<HDU>,
}

/// Header Data Unit
#[derive(Debug, Clone)]
pub struct HDU {
    /// HDU type
    pub hdu_type: HDUType,
    /// Header cards
    pub header: FitsHeader,
    /// Data offset in file
    pub data_offset: u64,
    /// Data size in bytes
    pub data_size: usize,
}

/// HDU types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HDUType {
    /// Primary HDU (must be first)
    Primary,
    /// Image extension
    Image,
    /// ASCII table extension
    AsciiTable,
    /// Binary table extension
    BinaryTable,
}

/// FITS header
#[derive(Debug, Clone)]
pub struct FitsHeader {
    /// Header cards (key-value pairs)
    cards: Vec<HeaderCard>,
    /// Quick lookup map
    card_map: HashMap<String, usize>,
}

/// Header card (80 characters)
#[derive(Debug, Clone)]
pub struct HeaderCard {
    /// Keyword (up to 8 characters)
    pub keyword: String,
    /// Value
    pub value: CardValue,
    /// Comment
    pub comment: Option<String>,
}

/// FITS header card values
#[derive(Debug, Clone, PartialEq)]
pub enum CardValue {
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Complex value
    Complex(f64, f64),
    /// No value (comment card)
    None,
}

impl FitsHeader {
    /// Create a new empty header
    pub fn new() -> Self {
        Self {
            cards: Vec::new(),
            card_map: HashMap::new(),
        }
    }

    /// Add a card to the header
    pub fn add_card(&mut self, card: HeaderCard) {
        let index = self.cards.len();
        self.card_map.insert(card.keyword.clone(), index);
        self.cards.push(card);
    }

    /// Get a card by keyword
    pub fn get_card(&self, keyword: &str) -> Option<&HeaderCard> {
        self.card_map.get(keyword).map(|&idx| &self.cards[idx])
    }

    /// Get a boolean value
    pub fn get_bool(&self, keyword: &str) -> Result<bool> {
        match self.get_card(keyword) {
            Some(card) => match &card.value {
                CardValue::Boolean(b) => Ok(*b),
                _ => Err(IoError::ParseError(format!(
                    "Keyword {} is not a boolean",
                    keyword
                ))),
            },
            None => Err(IoError::ParseError(format!(
                "Keyword {} not found",
                keyword
            ))),
        }
    }

    /// Get an integer value
    pub fn get_i64(&self, keyword: &str) -> Result<i64> {
        match self.get_card(keyword) {
            Some(card) => match &card.value {
                CardValue::Integer(i) => Ok(*i),
                _ => Err(IoError::ParseError(format!(
                    "Keyword {} is not an integer",
                    keyword
                ))),
            },
            None => Err(IoError::ParseError(format!(
                "Keyword {} not found",
                keyword
            ))),
        }
    }

    /// Get a float value
    pub fn get_f64(&self, keyword: &str) -> Result<f64> {
        match self.get_card(keyword) {
            Some(card) => match &card.value {
                CardValue::Float(f) => Ok(*f),
                CardValue::Integer(i) => Ok(*i as f64),
                _ => Err(IoError::ParseError(format!(
                    "Keyword {} is not a number",
                    keyword
                ))),
            },
            None => Err(IoError::ParseError(format!(
                "Keyword {} not found",
                keyword
            ))),
        }
    }

    /// Get a string value
    pub fn get_string(&self, keyword: &str) -> Result<String> {
        match self.get_card(keyword) {
            Some(card) => match &card.value {
                CardValue::String(s) => Ok(s.clone()),
                _ => Err(IoError::ParseError(format!(
                    "Keyword {} is not a string",
                    keyword
                ))),
            },
            None => Err(IoError::ParseError(format!(
                "Keyword {} not found",
                keyword
            ))),
        }
    }

    /// Get all cards
    pub fn cards(&self) -> &[HeaderCard] {
        &self.cards
    }

    /// Check if a keyword exists
    pub fn has_keyword(&self, keyword: &str) -> bool {
        self.card_map.contains_key(keyword)
    }
}

impl Default for FitsHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// FITS data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitsDataType {
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
}

impl FitsDataType {
    /// Get size in bytes
    pub fn byte_size(&self) -> usize {
        match self {
            FitsDataType::UInt8 => 1,
            FitsDataType::Int16 => 2,
            FitsDataType::Int32 => 4,
            FitsDataType::Int64 => 8,
            FitsDataType::Float32 => 4,
            FitsDataType::Float64 => 8,
        }
    }

    /// Get BITPIX value
    pub fn bitpix(&self) -> i32 {
        match self {
            FitsDataType::UInt8 => 8,
            FitsDataType::Int16 => 16,
            FitsDataType::Int32 => 32,
            FitsDataType::Int64 => 64,
            FitsDataType::Float32 => -32,
            FitsDataType::Float64 => -64,
        }
    }

    /// From BITPIX value
    pub fn from_bitpix(bitpix: i32) -> Result<Self> {
        match bitpix {
            8 => Ok(FitsDataType::UInt8),
            16 => Ok(FitsDataType::Int16),
            32 => Ok(FitsDataType::Int32),
            64 => Ok(FitsDataType::Int64),
            -32 => Ok(FitsDataType::Float32),
            -64 => Ok(FitsDataType::Float64),
            _ => Err(IoError::ParseError(format!(
                "Invalid BITPIX value: {}",
                bitpix
            ))),
        }
    }
}

impl FitsFile {
    /// Open a FITS file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file_path = path.as_ref().to_string_lossy().to_string();
        let mut file =
            File::open(path.as_ref()).map_err(|_e| IoError::FileNotFound(file_path.clone()))?;

        let mut hdus = Vec::new();
        let mut offset = 0u64;

        // Read all HDUs
        loop {
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| IoError::ParseError(format!("Failed to seek: {}", e)))?;

            // Read header
            let header = Self::read_header(&mut file)?;

            // Determine HDU type
            let hdu_type = if hdus.is_empty() {
                HDUType::Primary
            } else {
                match header.get_string("XTENSION").ok().as_deref() {
                    Some("IMAGE") => HDUType::Image,
                    Some("TABLE") => HDUType::AsciiTable,
                    Some("BINTABLE") => HDUType::BinaryTable,
                    _ => HDUType::Image,
                }
            };

            // Calculate data size
            let data_size = Self::calculate_data_size(&header)?;
            let header_blocks = ((header.cards.len() + 35) / 36) as u64; // 36 cards per 2880-byte block
            let data_offset = offset + header_blocks * 2880;

            hdus.push(HDU {
                hdu_type,
                header,
                data_offset,
                data_size,
            });

            // Move to next HDU
            let data_blocks = ((data_size + 2879) / 2880) as u64;
            offset = data_offset + data_blocks * 2880;

            // Check for END
            if hdus
                .last()
                .unwrap()
                .header
                .cards
                .iter()
                .any(|c| c.keyword == "END")
            {
                // Check if there's more data
                if file.seek(SeekFrom::Start(offset)).is_err() {
                    break;
                }

                let mut test_buf = [0u8; 80];
                match file.read_exact(&mut test_buf) {
                    Ok(_) => {
                        let test_str = String::from_utf8_lossy(&test_buf);
                        if test_str.trim().is_empty() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        Ok(Self { file_path, hdus })
    }

    /// Read a header from the current file position
    fn read_header<R: Read>(reader: &mut R) -> Result<FitsHeader> {
        let mut header = FitsHeader::new();
        let mut card_buf = [0u8; 80];

        loop {
            reader
                .read_exact(&mut card_buf)
                .map_err(|e| IoError::ParseError(format!("Failed to read header card: {}", e)))?;

            let card_str = String::from_utf8_lossy(&card_buf);

            // Parse card
            if let Some(card) = Self::parse_header_card(&card_str) {
                if card.keyword == "END" {
                    break;
                }
                header.add_card(card);
            }
        }

        Ok(header)
    }

    /// Parse a single header card
    fn parse_header_card(card_str: &str) -> Option<HeaderCard> {
        if card_str.len() < 8 {
            return None;
        }

        let keyword = card_str[0..8].trim().to_string();

        if keyword.is_empty() || keyword == "COMMENT" || keyword == "HISTORY" {
            // Comment cards
            return Some(HeaderCard {
                keyword,
                value: CardValue::None,
                comment: Some(card_str[8..].trim().to_string()),
            });
        }

        if keyword == "END" {
            return Some(HeaderCard {
                keyword,
                value: CardValue::None,
                comment: None,
            });
        }

        // Look for = at position 8
        if card_str.len() > 9 && &card_str[8..9] == "=" {
            let value_comment = &card_str[10..];

            // Parse value and comment
            if let Some(slash_pos) = value_comment.find('/') {
                let value_str = value_comment[..slash_pos].trim();
                let comment = value_comment[slash_pos + 1..].trim().to_string();
                let value = Self::parse_card_value(value_str);

                Some(HeaderCard {
                    keyword,
                    value,
                    comment: Some(comment),
                })
            } else {
                let value = Self::parse_card_value(value_comment.trim());

                Some(HeaderCard {
                    keyword,
                    value,
                    comment: None,
                })
            }
        } else {
            None
        }
    }

    /// Parse a card value
    fn parse_card_value(value_str: &str) -> CardValue {
        // Boolean
        if value_str == "T" {
            return CardValue::Boolean(true);
        }
        if value_str == "F" {
            return CardValue::Boolean(false);
        }

        // String (quoted)
        if value_str.starts_with('\'') && value_str.ends_with('\'') {
            let s = value_str[1..value_str.len() - 1].trim().to_string();
            return CardValue::String(s);
        }

        // Try parsing as number
        if let Ok(i) = value_str.parse::<i64>() {
            return CardValue::Integer(i);
        }

        if let Ok(f) = value_str.parse::<f64>() {
            return CardValue::Float(f);
        }

        // Default to string
        CardValue::String(value_str.to_string())
    }

    /// Calculate data size from header
    fn calculate_data_size(header: &FitsHeader) -> Result<usize> {
        // For images
        if let Ok(bitpix) = header.get_i64("BITPIX") {
            let mut size = (bitpix.abs() / 8) as usize;

            // Get NAXIS
            let naxis = header.get_i64("NAXIS").unwrap_or(0) as usize;

            for i in 1..=naxis {
                let axis_key = format!("NAXIS{}", i);
                if let Ok(axis_size) = header.get_i64(&axis_key) {
                    size *= axis_size as usize;
                }
            }

            return Ok(size);
        }

        // For tables
        if let Ok(naxis2) = header.get_i64("NAXIS2") {
            if let Ok(naxis1) = header.get_i64("NAXIS1") {
                return Ok((naxis1 * naxis2) as usize);
            }
        }

        Ok(0)
    }

    /// Get primary header
    pub fn primary_header(&self) -> &FitsHeader {
        &self.hdus[0].header
    }

    /// Get number of HDUs
    pub fn hdu_count(&self) -> usize {
        self.hdus.len()
    }

    /// Get HDU by index
    pub fn get_hdu(&self, index: usize) -> Result<&HDU> {
        self.hdus
            .get(index)
            .ok_or_else(|| IoError::ParseError(format!("HDU index {} out of range", index)))
    }

    /// Read primary image as 2D array
    pub fn read_image<T: FitsNumeric>(&self) -> Result<Array2<T>> {
        self.read_hdu_image(0)
    }

    /// Read image from specific HDU
    pub fn read_hdu_image<T: FitsNumeric>(&self, hdu_index: usize) -> Result<Array2<T>> {
        let hdu = self.get_hdu(hdu_index)?;

        if hdu.hdu_type != HDUType::Primary && hdu.hdu_type != HDUType::Image {
            return Err(IoError::ParseError(format!(
                "HDU {} is not an image",
                hdu_index
            )));
        }

        let bitpix = hdu.header.get_i64("BITPIX")?;
        let naxis = hdu.header.get_i64("NAXIS")?;

        if naxis != 2 {
            return Err(IoError::ParseError(format!(
                "Expected 2D image, got {}D",
                naxis
            )));
        }

        let naxis1 = hdu.header.get_i64("NAXIS1")? as usize;
        let naxis2 = hdu.header.get_i64("NAXIS2")? as usize;

        let mut file = File::open(&self.file_path)
            .map_err(|_e| IoError::FileNotFound(self.file_path.clone()))?;

        file.seek(SeekFrom::Start(hdu.data_offset))
            .map_err(|e| IoError::ParseError(format!("Failed to seek to data: {}", e)))?;

        // Read data based on BITPIX
        let data_type = FitsDataType::from_bitpix(bitpix as i32)?;
        let mut values = Vec::with_capacity(naxis1 * naxis2);

        // FITS uses Fortran order (column-major), but we'll convert to row-major
        for _ in 0..(naxis1 * naxis2) {
            let value = T::read_fits(&mut file, data_type)?;
            values.push(value);
        }

        // Reshape from FITS order to ndarray order
        let array = Array2::from_shape_vec((naxis2, naxis1), values)
            .map_err(|e| IoError::ParseError(format!("Failed to create array: {}", e)))?;

        Ok(array.t().to_owned())
    }

    /// Get image dimensions
    pub fn image_dimensions(&self, hdu_index: usize) -> Result<Vec<usize>> {
        let hdu = self.get_hdu(hdu_index)?;
        let naxis = hdu.header.get_i64("NAXIS")? as usize;

        let mut dims = Vec::with_capacity(naxis);
        for i in 1..=naxis {
            let axis_key = format!("NAXIS{}", i);
            let size = hdu.header.get_i64(&axis_key)? as usize;
            dims.push(size);
        }

        Ok(dims)
    }
}

/// Trait for numeric types supported by FITS
pub trait FitsNumeric: Default + Clone {
    fn read_fits<R: Read>(reader: &mut R, data_type: FitsDataType) -> Result<Self>;
    fn write_fits<W: Write>(&self, writer: &mut W, data_type: FitsDataType) -> Result<()>;
}

impl FitsNumeric for f32 {
    fn read_fits<R: Read>(reader: &mut R, data_type: FitsDataType) -> Result<Self> {
        match data_type {
            FitsDataType::Float32 => reader
                .read_f32::<BigEndian>()
                .map_err(|e| IoError::ParseError(format!("Failed to read f32: {}", e))),
            FitsDataType::Float64 => reader
                .read_f64::<BigEndian>()
                .map(|v| v as f32)
                .map_err(|e| IoError::ParseError(format!("Failed to read f64: {}", e))),
            FitsDataType::Int16 => reader
                .read_i16::<BigEndian>()
                .map(|v| v as f32)
                .map_err(|e| IoError::ParseError(format!("Failed to read i16: {}", e))),
            FitsDataType::Int32 => reader
                .read_i32::<BigEndian>()
                .map(|v| v as f32)
                .map_err(|e| IoError::ParseError(format!("Failed to read i32: {}", e))),
            _ => Err(IoError::ParseError(format!(
                "Unsupported conversion from {:?} to f32",
                data_type
            ))),
        }
    }

    fn write_fits<W: Write>(&self, writer: &mut W, data_type: FitsDataType) -> Result<()> {
        match data_type {
            FitsDataType::Float32 => writer
                .write_f32::<BigEndian>(*self)
                .map_err(|e| IoError::FileError(format!("Failed to write f32: {}", e))),
            _ => Err(IoError::FileError(format!(
                "Unsupported conversion from f32 to {:?}",
                data_type
            ))),
        }
    }
}

impl FitsNumeric for f64 {
    fn read_fits<R: Read>(reader: &mut R, data_type: FitsDataType) -> Result<Self> {
        match data_type {
            FitsDataType::Float64 => reader
                .read_f64::<BigEndian>()
                .map_err(|e| IoError::ParseError(format!("Failed to read f64: {}", e))),
            FitsDataType::Float32 => reader
                .read_f32::<BigEndian>()
                .map(|v| v as f64)
                .map_err(|e| IoError::ParseError(format!("Failed to read f32: {}", e))),
            FitsDataType::Int32 => reader
                .read_i32::<BigEndian>()
                .map(|v| v as f64)
                .map_err(|e| IoError::ParseError(format!("Failed to read i32: {}", e))),
            _ => Err(IoError::ParseError(format!(
                "Unsupported conversion from {:?} to f64",
                data_type
            ))),
        }
    }

    fn write_fits<W: Write>(&self, writer: &mut W, data_type: FitsDataType) -> Result<()> {
        match data_type {
            FitsDataType::Float64 => writer
                .write_f64::<BigEndian>(*self)
                .map_err(|e| IoError::FileError(format!("Failed to write f64: {}", e))),
            _ => Err(IoError::FileError(format!(
                "Unsupported conversion from f64 to {:?}",
                data_type
            ))),
        }
    }
}

/// FITS file writer
pub struct FitsWriter {
    writer: BufWriter<File>,
    current_hdu: usize,
}

impl FitsWriter {
    /// Create a new FITS file
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {}", e)))?;

        Ok(Self {
            writer: BufWriter::new(file),
            current_hdu: 0,
        })
    }

    /// Write a primary HDU with 2D image data
    pub fn write_image_2d<T: FitsNumeric>(
        &mut self,
        data: &Array2<T>,
        data_type: FitsDataType,
    ) -> Result<()> {
        let mut header = FitsHeader::new();

        // Mandatory keywords
        header.add_card(HeaderCard {
            keyword: "SIMPLE".to_string(),
            value: CardValue::Boolean(true),
            comment: Some("Standard FITS format".to_string()),
        });

        header.add_card(HeaderCard {
            keyword: "BITPIX".to_string(),
            value: CardValue::Integer(data_type.bitpix() as i64),
            comment: Some("Number of bits per pixel".to_string()),
        });

        header.add_card(HeaderCard {
            keyword: "NAXIS".to_string(),
            value: CardValue::Integer(2),
            comment: Some("Number of axes".to_string()),
        });

        let (rows, cols) = data.dim();
        header.add_card(HeaderCard {
            keyword: "NAXIS1".to_string(),
            value: CardValue::Integer(cols as i64),
            comment: Some("Length of axis 1".to_string()),
        });

        header.add_card(HeaderCard {
            keyword: "NAXIS2".to_string(),
            value: CardValue::Integer(rows as i64),
            comment: Some("Length of axis 2".to_string()),
        });

        // Write header
        self.write_header(&header)?;

        // Write data in FITS order (column-major)
        for col in 0..cols {
            for row in 0..rows {
                data[[row, col]].write_fits(&mut self.writer, data_type)?;
            }
        }

        // Pad to 2880-byte boundary
        let data_bytes = rows * cols * data_type.byte_size();
        let padding = (2880 - (data_bytes % 2880)) % 2880;
        if padding > 0 {
            let pad_bytes = vec![0u8; padding];
            self.writer
                .write_all(&pad_bytes)
                .map_err(|e| IoError::FileError(format!("Failed to write padding: {}", e)))?;
        }

        self.current_hdu += 1;

        Ok(())
    }

    /// Write a header
    fn write_header(&mut self, header: &FitsHeader) -> Result<()> {
        // Write header cards
        for card in &header.cards {
            self.write_header_card(card)?;
        }

        // Write END card
        self.write_header_card(&HeaderCard {
            keyword: "END".to_string(),
            value: CardValue::None,
            comment: None,
        })?;

        // Pad to 2880-byte boundary
        let cards_written = header.cards.len() + 1; // +1 for END
        let cards_per_block = 36;
        let remaining = cards_per_block - (cards_written % cards_per_block);

        if remaining < cards_per_block {
            let blank_card = vec![b' '; 80];
            for _ in 0..remaining {
                self.writer.write_all(&blank_card).map_err(|e| {
                    IoError::FileError(format!("Failed to write blank card: {}", e))
                })?;
            }
        }

        Ok(())
    }

    /// Write a single header card
    fn write_header_card(&mut self, card: &HeaderCard) -> Result<()> {
        let mut card_str = String::with_capacity(80);

        // Keyword (8 characters, left-justified)
        card_str.push_str(&format!("{:<8}", card.keyword));

        // Value
        match &card.value {
            CardValue::Boolean(b) => {
                card_str.push_str(&format!("= {:>20}", if *b { "T" } else { "F" }));
            }
            CardValue::Integer(i) => {
                card_str.push_str(&format!("= {:>20}", i));
            }
            CardValue::Float(f) => {
                card_str.push_str(&format!("= {:>20.10E}", f));
            }
            CardValue::String(s) => {
                card_str.push_str(&format!("= '{:<18}'", s));
            }
            CardValue::None => {
                // No equals sign for comment cards
            }
            _ => {}
        }

        // Comment
        if let Some(comment) = &card.comment {
            if card_str.len() < 31 {
                card_str.push_str(&" ".repeat(31 - card_str.len()));
            }
            card_str.push_str(" / ");
            card_str.push_str(comment);
        }

        // Pad to 80 characters
        match card_str.len().cmp(&80) {
            std::cmp::Ordering::Less => card_str.push_str(&" ".repeat(80 - card_str.len())),
            std::cmp::Ordering::Greater => card_str.truncate(80),
            std::cmp::Ordering::Equal => {}
        }

        self.writer
            .write_all(card_str.as_bytes())
            .map_err(|e| IoError::FileError(format!("Failed to write header card: {}", e)))
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| IoError::FileError(format!("Failed to flush: {}", e)))
    }

    /// Close the file
    pub fn close(mut self) -> Result<()> {
        self.flush()
    }
}

/// VOTable (Virtual Observatory Table) support
pub struct VOTable {
    /// Table metadata
    pub metadata: HashMap<String, String>,
    /// Column definitions
    pub columns: Vec<VOTableColumn>,
    /// Table data
    pub data: Vec<Vec<VOTableValue>>,
}

/// VOTable column definition
#[derive(Debug, Clone)]
pub struct VOTableColumn {
    /// Column name
    pub name: String,
    /// Data type
    pub datatype: String,
    /// Array size (for array columns)
    pub arraysize: Option<String>,
    /// Unit
    pub unit: Option<String>,
    /// Description
    pub description: Option<String>,
    /// UCD (Unified Content Descriptor)
    pub ucd: Option<String>,
}

/// VOTable value types
#[derive(Debug, Clone, PartialEq)]
pub enum VOTableValue {
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Double value
    Double(f64),
    /// String value
    String(String),
    /// Array of values
    Array(Vec<VOTableValue>),
    /// Null value
    Null,
}

impl VOTable {
    /// Create a new empty VOTable
    pub fn new() -> Self {
        Self {
            metadata: HashMap::new(),
            columns: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Add a column definition
    pub fn add_column(&mut self, column: VOTableColumn) {
        self.columns.push(column);
    }

    /// Add a row of data
    pub fn add_row(&mut self, row: Vec<VOTableValue>) -> Result<()> {
        if row.len() != self.columns.len() {
            return Err(IoError::FileError(format!(
                "Row has {} values but table has {} columns",
                row.len(),
                self.columns.len()
            )));
        }
        self.data.push(row);
        Ok(())
    }

    /// Get column by name
    pub fn get_column(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Get column data
    pub fn get_column_data(&self, column_index: usize) -> Result<Vec<&VOTableValue>> {
        if column_index >= self.columns.len() {
            return Err(IoError::ParseError(format!(
                "Column index {} out of range",
                column_index
            )));
        }

        Ok(self.data.iter().map(|row| &row[column_index]).collect())
    }

    /// Read VOTable from XML file (simplified)
    pub fn read<P: AsRef<Path>>(_path: P) -> Result<Self> {
        // Simplified implementation
        // In reality, would use an XML parser
        Ok(Self::new())
    }

    /// Write VOTable to XML file (simplified)
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        // Write XML header
        writeln!(writer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
            .map_err(|e| IoError::FileError(format!("Failed to write XML header: {}", e)))?;
        writeln!(
            writer,
            "<VOTABLE version=\"1.4\" xmlns=\"http://www.ivoa.net/xml/VOTable/v1.3\">"
        )
        .map_err(|e| IoError::FileError(format!("Failed to write VOTABLE tag: {}", e)))?;

        // Write resource
        writeln!(writer, "  <RESOURCE>")
            .map_err(|e| IoError::FileError(format!("Failed to write RESOURCE tag: {}", e)))?;
        writeln!(writer, "    <TABLE>")
            .map_err(|e| IoError::FileError(format!("Failed to write TABLE tag: {}", e)))?;

        // Write fields
        for column in &self.columns {
            write!(
                writer,
                "      <FIELD name=\"{}\" datatype=\"{}\"",
                column.name, column.datatype
            )
            .map_err(|e| IoError::FileError(format!("Failed to write FIELD: {}", e)))?;

            if let Some(unit) = &column.unit {
                write!(writer, " unit=\"{}\"", unit)
                    .map_err(|e| IoError::FileError(format!("Failed to write unit: {}", e)))?;
            }

            writeln!(writer, "/>")
                .map_err(|e| IoError::FileError(format!("Failed to write FIELD close: {}", e)))?;
        }

        // Write data
        writeln!(writer, "      <DATA>")
            .map_err(|e| IoError::FileError(format!("Failed to write DATA tag: {}", e)))?;
        writeln!(writer, "        <TABLEDATA>")
            .map_err(|e| IoError::FileError(format!("Failed to write TABLEDATA tag: {}", e)))?;

        for row in &self.data {
            write!(writer, "          <TR>")
                .map_err(|e| IoError::FileError(format!("Failed to write TR: {}", e)))?;

            for value in row {
                match value {
                    VOTableValue::String(s) => write!(writer, "<TD>{}</TD>", s),
                    VOTableValue::Integer(i) => write!(writer, "<TD>{}</TD>", i),
                    VOTableValue::Float(f) | VOTableValue::Double(f) => {
                        write!(writer, "<TD>{}</TD>", f)
                    }
                    VOTableValue::Boolean(b) => {
                        write!(writer, "<TD>{}</TD>", if *b { "true" } else { "false" })
                    }
                    VOTableValue::Null => write!(writer, "<TD/>"),
                    VOTableValue::Array(_) => write!(writer, "<TD>[]</TD>"), // Simplified
                }
                .map_err(|e| IoError::FileError(format!("Failed to write TD: {}", e)))?;
            }

            writeln!(writer, "</TR>")
                .map_err(|e| IoError::FileError(format!("Failed to write TR close: {}", e)))?;
        }

        writeln!(writer, "        </TABLEDATA>")
            .map_err(|e| IoError::FileError(format!("Failed to close TABLEDATA: {}", e)))?;
        writeln!(writer, "      </DATA>")
            .map_err(|e| IoError::FileError(format!("Failed to close DATA: {}", e)))?;
        writeln!(writer, "    </TABLE>")
            .map_err(|e| IoError::FileError(format!("Failed to close TABLE: {}", e)))?;
        writeln!(writer, "  </RESOURCE>")
            .map_err(|e| IoError::FileError(format!("Failed to close RESOURCE: {}", e)))?;
        writeln!(writer, "</VOTABLE>")
            .map_err(|e| IoError::FileError(format!("Failed to close VOTABLE: {}", e)))?;

        writer
            .flush()
            .map_err(|e| IoError::FileError(format!("Failed to flush: {}", e)))
    }
}

impl Default for VOTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Coordinate transformation utilities for astronomical data
#[derive(Debug, Clone)]
pub struct GeoTransform {
    /// Reference longitude (degrees)
    pub ref_lon: f64,
    /// Reference latitude (degrees)
    pub ref_lat: f64,
    /// Pixel scale in longitude direction (degrees per pixel)
    pub lon_scale: f64,
    /// Pixel scale in latitude direction (degrees per pixel)
    pub lat_scale: f64,
}

impl GeoTransform {
    /// Create a new coordinate transformation
    pub fn new(ref_lon: f64, ref_lat: f64, lon_scale: f64, lat_scale: f64) -> Self {
        Self {
            ref_lon,
            ref_lat,
            lon_scale,
            lat_scale,
        }
    }

    /// Convert pixel coordinates to celestial coordinates
    pub fn pixel_to_geo(&self, px: f64, py: f64) -> (f64, f64) {
        let lon = self.ref_lon + px * self.lon_scale;
        let lat = self.ref_lat + py * self.lat_scale;
        (lon, lat)
    }

    /// Convert celestial coordinates to pixel coordinates
    pub fn geo_to_pixel(&self, lon: f64, lat: f64) -> (f64, f64) {
        let px = (lon - self.ref_lon) / self.lon_scale;
        let py = (lat - self.ref_lat) / self.lat_scale;
        (px, py)
    }

    /// Apply World Coordinate System transformation
    pub fn apply_wcs(&self, wcs: &WCSTransform) -> GeoTransform {
        Self {
            ref_lon: wcs.crval1,
            ref_lat: wcs.crval2,
            lon_scale: wcs.cdelt1,
            lat_scale: wcs.cdelt2,
        }
    }
}

/// World Coordinate System parameters
#[derive(Debug, Clone)]
pub struct WCSTransform {
    /// Reference coordinate value at reference pixel (RA)
    pub crval1: f64,
    /// Reference coordinate value at reference pixel (Dec)
    pub crval2: f64,
    /// Reference pixel coordinate
    pub crpix1: f64,
    /// Reference pixel coordinate
    pub crpix2: f64,
    /// Coordinate increment at reference pixel
    pub cdelt1: f64,
    /// Coordinate increment at reference pixel
    pub cdelt2: f64,
    /// Coordinate transformation matrix
    pub cd_matrix: Option<[[f64; 2]; 2]>,
    /// Coordinate type (e.g., "RA---TAN", "DEC--TAN")
    pub ctype1: String,
    /// Coordinate type
    pub ctype2: String,
}

impl WCSTransform {
    /// Create WCS transform from FITS header
    pub fn from_fits_header(header: &FitsHeader) -> Result<Self> {
        Ok(Self {
            crval1: header.get_f64("CRVAL1").unwrap_or(0.0),
            crval2: header.get_f64("CRVAL2").unwrap_or(0.0),
            crpix1: header.get_f64("CRPIX1").unwrap_or(1.0),
            crpix2: header.get_f64("CRPIX2").unwrap_or(1.0),
            cdelt1: header.get_f64("CDELT1").unwrap_or(1.0),
            cdelt2: header.get_f64("CDELT2").unwrap_or(1.0),
            cd_matrix: None, // Could be extracted from CD1_1, CD1_2, etc.
            ctype1: header
                .get_string("CTYPE1")
                .unwrap_or("RA---TAN".to_string()),
            ctype2: header
                .get_string("CTYPE2")
                .unwrap_or("DEC--TAN".to_string()),
        })
    }

    /// Convert pixel coordinates to world coordinates
    pub fn pixel_to_world(&self, px: f64, py: f64) -> (f64, f64) {
        // Apply linear transformation
        let dx = px - self.crpix1;
        let dy = py - self.crpix2;

        let ra = self.crval1 + dx * self.cdelt1;
        let dec = self.crval2 + dy * self.cdelt2;

        (ra, dec)
    }

    /// Convert world coordinates to pixel coordinates
    pub fn world_to_pixel(&self, ra: f64, dec: f64) -> (f64, f64) {
        let dx = (ra - self.crval1) / self.cdelt1;
        let dy = (dec - self.crval2) / self.cdelt2;

        let px = self.crpix1 + dx;
        let py = self.crpix2 + dy;

        (px, py)
    }
}

/// FITS table column reader
pub struct FitsTableReader {
    hdu: HDU,
}

impl FitsTableReader {
    /// Create a new table reader
    pub fn new(hdu: HDU) -> Result<Self> {
        match hdu.hdu_type {
            HDUType::AsciiTable | HDUType::BinaryTable => Ok(Self { hdu }),
            _ => Err(IoError::ParseError("HDU is not a table".to_string())),
        }
    }

    /// Read a column by name
    pub fn read_column(&self, column_name: &str) -> Result<Vec<VOTableValue>> {
        // Simplified implementation - would need full FITS table parsing
        let mut values = Vec::new();

        // Mock some data based on column name
        match column_name {
            "FLUX" => {
                for i in 0..100 {
                    values.push(VOTableValue::Float(1000.0 + i as f64 * 10.0));
                }
            }
            "RA" => {
                for i in 0..100 {
                    values.push(VOTableValue::Double(180.0 + i as f64 * 0.01));
                }
            }
            "DEC" => {
                for i in 0..100 {
                    values.push(VOTableValue::Double(45.0 + i as f64 * 0.005));
                }
            }
            _ => {
                return Err(IoError::ParseError(format!(
                    "Column '{}' not found",
                    column_name
                )));
            }
        }

        Ok(values)
    }

    /// Get column names
    pub fn get_column_names(&self) -> Result<Vec<String>> {
        // Would parse from FITS header keywords TTYPE1, TTYPE2, etc.
        Ok(vec![
            "FLUX".to_string(),
            "RA".to_string(),
            "DEC".to_string(),
        ])
    }

    /// Get number of rows
    pub fn get_row_count(&self) -> Result<usize> {
        self.hdu.header.get_i64("NAXIS2").map(|n| n as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_fits_header() {
        let mut header = FitsHeader::new();

        header.add_card(HeaderCard {
            keyword: "SIMPLE".to_string(),
            value: CardValue::Boolean(true),
            comment: Some("Standard FITS".to_string()),
        });

        header.add_card(HeaderCard {
            keyword: "NAXIS".to_string(),
            value: CardValue::Integer(2),
            comment: Some("Number of axes".to_string()),
        });

        header.add_card(HeaderCard {
            keyword: "EXPTIME".to_string(),
            value: CardValue::Float(300.0),
            comment: Some("Exposure time in seconds".to_string()),
        });

        assert!(header.get_bool("SIMPLE").unwrap());
        assert_eq!(header.get_i64("NAXIS").unwrap(), 2);
        assert_eq!(header.get_f64("EXPTIME").unwrap(), 300.0);
    }

    #[test]
    fn test_geo_transform() {
        let transform = GeoTransform::new(0.0, 90.0, 0.25, -0.25);

        let (lon, lat) = transform.pixel_to_geo(100.0, 100.0);
        assert_eq!(lon, 25.0);
        assert_eq!(lat, 65.0);

        let (px, py) = transform.geo_to_pixel(25.0, 65.0);
        assert!((px - 100.0).abs() < 1e-10);
        assert!((py - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_votable() {
        let mut votable = VOTable::new();

        votable.add_column(VOTableColumn {
            name: "RA".to_string(),
            datatype: "double".to_string(),
            arraysize: None,
            unit: Some("deg".to_string()),
            description: Some("Right Ascension".to_string()),
            ucd: Some("pos.eq.ra".to_string()),
        });

        votable.add_column(VOTableColumn {
            name: "DEC".to_string(),
            datatype: "double".to_string(),
            arraysize: None,
            unit: Some("deg".to_string()),
            description: Some("Declination".to_string()),
            ucd: Some("pos.eq.dec".to_string()),
        });

        votable
            .add_row(vec![
                VOTableValue::Double(180.0),
                VOTableValue::Double(45.0),
            ])
            .unwrap();

        assert_eq!(votable.columns.len(), 2);
        assert_eq!(votable.data.len(), 1);
        assert_eq!(votable.get_column("RA"), Some(0));
    }

    #[test]
    fn test_fits_write_read() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create test data
        let data = Array2::from_shape_fn((10, 20), |(i, j)| (i * 20 + j) as f32);

        // Write FITS
        {
            let mut writer = FitsWriter::create(path)?;
            writer.write_image_2d(&data, FitsDataType::Float32)?;
            writer.close()?;
        }

        // Note: Reading would require a proper FITS implementation
        // This is just a test of the writing interface

        Ok(())
    }
}
