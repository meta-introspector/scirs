//! Slicing operations for memory-mapped arrays.
//!
//! This module provides functionality for efficiently slicing memory-mapped arrays
//! without loading the entire array into memory. These slicing operations maintain
//! the memory-mapping and only load the required data when accessed.

#[cfg(test)]
use super::memmap::AccessMode;
use super::memmap::MemoryMappedArray;
use crate::error::{CoreError, CoreResult, ErrorContext};
use ndarray::{ArrayBase, Dimension, IxDyn, SliceInfo, SliceInfoElem};
use std::marker::PhantomData;
use std::ops::RangeBounds;

/// A slice of a memory-mapped array that maintains memory-mapping.
///
/// This provides a view into a subset of a memory-mapped array without
/// loading the entire array into memory. Data is only loaded when
/// accessed through the slice.
pub struct MemoryMappedSlice<A, D>
where
    A: Clone + Copy + 'static + Send + Sync,
    D: Dimension,
{
    /// The source memory-mapped array
    source: MemoryMappedArray<A>,

    /// The slice information
    slice_info: SliceInfo<Vec<SliceInfoElem>, D, D>,

    /// Phantom data for dimension type
    _phantom: PhantomData<D>,
}

impl<A, D> MemoryMappedSlice<A, D>
where
    A: Clone + Copy + 'static + Send + Sync,
    D: Dimension,
{
    /// Creates a new slice from a memory-mapped array and slice information.
    pub fn new(
        source: MemoryMappedArray<A>,
        slice_info: SliceInfo<Vec<SliceInfoElem>, D, D>,
    ) -> Self {
        Self {
            source,
            slice_info,
            _phantom: PhantomData,
        }
    }

    /// Returns the shape of the slice.
    ///
    /// Since we can't directly access the private out_dim field in SliceInfo,
    /// this just returns an empty dimension. Actual implementations would
    /// need to calculate this based on the slice parameters.
    pub fn shape(&self) -> D {
        D::default()
    }

    /// Returns a reference to the source memory-mapped array.
    pub const fn source(&self) -> &MemoryMappedArray<A> {
        &self.source
    }

    /// Returns the slice information.
    pub const fn slice_info(&self) -> &SliceInfo<Vec<SliceInfoElem>, D, D> {
        &self.slice_info
    }

    /// Safely convert an array to the target dimension type with detailed error reporting.
    fn safe_dimensionality_conversion(
        array: ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::IxDyn>,
        context: &str,
    ) -> CoreResult<ArrayBase<ndarray::OwnedRepr<A>, D>> {
        let source_shape = array.shape().to_vec();
        let source_ndim = source_shape.len();
        let target_ndim = D::NDIM;

        // Try direct conversion first
        match array.into_dimensionality::<D>() {
            Ok(converted) => Ok(converted),
            Err(_original_array) => {
                // Conversion failed, try to provide helpful error message and fallback strategies
                let error_msg = match (source_ndim, target_ndim) {
                    (s, Some(t)) if s == t => {
                        format!(
                            "Dimension conversion failed for {} array despite matching dimensions ({} -> {}). \
                             Source shape: {:?}, target dimension type: {}",
                            context, s, t, source_shape, std::any::type_name::<D>()
                        )
                    }
                    (s, Some(t)) if s > t => {
                        format!(
                            "Cannot convert {} array: too many dimensions ({} -> {}). \
                             Source shape: {:?}. Consider using a higher-dimensional target type or \
                             applying additional slicing to reduce dimensions.",
                            context, s, t, source_shape
                        )
                    }
                    (s, Some(t)) if s < t => {
                        // Try to add singleton dimensions for lower-dimensional arrays
                        // Cannot expand dimensions automatically - return error
                        format!(
                            "Cannot expand {} array from {} to {} dimensions automatically. \
                             Source shape: {:?}. Consider reshaping the array manually.",
                            context, s, t, source_shape
                        )
                    }
                    (s, None) => {
                        format!(
                            "Cannot convert {} array to dynamic dimension type. \
                             Source shape: {:?}, source dimensions: {}",
                            context, source_shape, s
                        )
                    }
                    _ => {
                        format!(
                            "Unexpected dimension conversion failure for {} array. \
                             Source shape: {:?}",
                            context, source_shape
                        )
                    }
                };

                Err(CoreError::ShapeError(ErrorContext::new(error_msg)))
            }
        }
    }

    /// Try to expand dimensions by adding singleton dimensions.
    fn try_expand_dimensions(
        array: ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::IxDyn>,
        context: &str,
        source_dims: usize,
        target_dims: usize,
    ) -> CoreResult<ArrayBase<ndarray::OwnedRepr<A>, D>> {
        // For cases where we have fewer dimensions than needed, try adding singleton dimensions
        if target_dims == 2 && source_dims == 1 {
            // Try to reshape 1D array to 2D by adding a singleton dimension
            let len = array.len();

            // Try (len, 1) first
            if let Ok(reshaped) = array.clone().into_shape_with_order((len, 1)) {
                if let Ok(converted) = reshaped.into_dimensionality::<D>() {
                    return Ok(converted);
                }
            }

            // Try (1, len) orientation
            if let Ok(reshaped) = array.into_shape_with_order((1, len)) {
                if let Ok(converted) = reshaped.into_dimensionality::<D>() {
                    return Ok(converted);
                }
            }
        }

        Err(CoreError::ShapeError(ErrorContext::new(format!(
            "Cannot expand {} array from {} to {} dimensions. \
             Automatic dimension expansion failed.",
            context, source_dims, target_dims
        ))))
    }

    /// Loads the slice data into memory.
    ///
    /// This method materializes the slice by loading only the necessary data
    /// from the memory-mapped file.
    pub fn load(&self) -> CoreResult<ArrayBase<ndarray::OwnedRepr<A>, D>> {
        // Get the raw data slice
        let data_slice = self.source.as_slice();

        // Use generic approach that works for all dimension types
        self.load_slice_generic(data_slice)
    }

    /// Generic slice loading that works for all dimension types
    fn load_slice_generic(&self, data_slice: &[A]) -> CoreResult<ArrayBase<ndarray::OwnedRepr<A>, D>> {
        use ndarray::IxDyn;
        
        // Validate dimension compatibility first
        self.validate_dimension_compatibility()?;
        
        // Create dynamic array view from source
        let source_shape = IxDyn(&self.source.shape);
        let source_array = ndarray::ArrayView::from_shape(source_shape, data_slice).map_err(|e| {
            CoreError::ShapeError(ErrorContext::new(format!(
                "Failed to create array view from source shape {:?}: {}",
                self.source.shape, e
            )))
        })?;

        // Apply the slice using ndarray's generic slicing
        let slice_elements = self.slice_info.as_ref();
        let sliced = self.apply_slice_safely_owned(source_array, slice_elements)?;

        // Convert to target dimension with robust error handling
        Self::safe_dimensionality_conversion(sliced, "sliced array")
    }

    /// Validate that the slice operation is compatible with target dimension
    fn validate_dimension_compatibility(&self) -> CoreResult<()> {
        let source_ndim = self.source.shape.len();
        let slice_elements = self.slice_info.as_ref();
        
        // Count how many dimensions will remain after slicing (non-index slices)
        let remaining_dims = slice_elements.iter()
            .take(source_ndim) // Only count up to source dimensions
            .filter(|elem| !matches!(elem, SliceInfoElem::Index(_)))
            .count()
            .max(source_ndim - slice_elements.iter()
                .filter(|elem| matches!(elem, SliceInfoElem::Index(_)))
                .count());

        // Check if target dimension is compatible
        if let Some(target_ndim) = D::NDIM {
            if remaining_dims != target_ndim {
                return Err(CoreError::DimensionError(ErrorContext::new(format!(
                    "Dimension mismatch: slice operation will result in {}D array, but target type expects {}D. \
                     Source shape: {:?}, slice elements: {} (including {} index operations)",
                    remaining_dims, target_ndim, self.source.shape, slice_elements.len(),
                    slice_elements.iter().filter(|e| matches!(e, SliceInfoElem::Index(_))).count()
                ))));
            }
        }

        Ok(())
    }

    /// Safely apply slice to array view with proper error handling, returning owned array
    fn apply_slice_safely_owned(
        &self,
        source_array: ndarray::ArrayView<A, IxDyn>,
        slice_elements: &[SliceInfoElem],
    ) -> CoreResult<ndarray::Array<A, IxDyn>> {
        if slice_elements.is_empty() {
            return Ok(source_array.to_owned());
        }

        // Apply the slice using ndarray's slicing
        let sliced = source_array.slice_each_axis(|ax| {
            if ax.axis.index() < slice_elements.len() {
                match &slice_elements[ax.axis.index()] {
                    SliceInfoElem::Slice { start, end, step } => {
                        // Handle negative indices and bounds checking
                        let dim_size = ax.len as isize;
                        let safe_start = self.handle_negative_index(*start, dim_size);
                        let safe_end = if let Some(e) = end {
                            self.handle_negative_index(*e, dim_size)
                        } else {
                            dim_size
                        };
                        
                        // Ensure indices are within bounds
                        let clamped_start = safe_start.max(0).min(dim_size) as usize;
                        let clamped_end = safe_end.max(0).min(dim_size) as usize;
                        
                        // Validate step
                        let safe_step = step.max(&1).abs() as usize;
                        
                        ndarray::Slice::new(clamped_start as isize, Some(clamped_end as isize), safe_step as isize)
                    }
                    SliceInfoElem::Index(idx) => {
                        let dim_size = ax.len as isize;
                        let safe_idx = self.handle_negative_index(*idx, dim_size);
                        let clamped_idx = safe_idx.max(0).min(dim_size - 1) as usize;
                        ndarray::Slice::new(clamped_idx as isize, Some((clamped_idx + 1) as isize), 1)
                    }
                    _ => ndarray::Slice::new(0, None, 1),
                }
            } else {
                ndarray::Slice::new(0, None, 1)
            }
        });

        Ok(sliced.to_owned())
    }

    /// Handle negative indices properly  
    fn handle_negative_index(&self, index: isize, dim_size: isize) -> isize {
        if index < 0 {
            dim_size + index
        } else {
            index
        }
    }
}

/// Extension trait for adding slicing functionality to MemoryMappedArray.
pub trait MemoryMappedSlicing<A: Clone + Copy + 'static + Send + Sync> {
    /// Creates a slice of the memory-mapped array using standard slice syntax.
    fn slice<I, E>(&self, _info: I) -> CoreResult<MemoryMappedSlice<A, E>>
    where
        I: ndarray::SliceArg<E>,
        E: Dimension;

    /// Creates a 1D slice using a range.
    fn slice_1d(
        &self,
        range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, ndarray::Ix1>>;

    /// Creates a 2D slice using ranges for each dimension.
    fn slice_2d(
        &self,
        row_range: impl RangeBounds<usize>,
        col_range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, ndarray::Ix2>>;
}

impl<A: Clone + Copy + 'static + Send + Sync> MemoryMappedSlicing<A> for MemoryMappedArray<A> {
    fn slice<I, E>(&self, _info: I) -> CoreResult<MemoryMappedSlice<A, E>>
    where
        I: ndarray::SliceArg<E>,
        E: Dimension,
    {
        // For now, we'll implement specific cases and improve later
        // This is a limitation of the current API

        // Create a default slice that returns the whole array
        // This is a limitation - we can't properly convert generic SliceArg to SliceInfo
        // without knowing the specific slice type at compile time
        let sliced_shape = self.shape.clone();

        // Create SliceInfo that represents the identity slice on the sliced data
        // This is because we're creating a new MemoryMappedArray that contains just the sliced data
        let mut elems = Vec::new();
        for &dim_size in &sliced_shape {
            elems.push(SliceInfoElem::Slice {
                start: 0,
                end: Some(dim_size as isize),
                step: 1,
            });
        }

        let slice_info = unsafe { SliceInfo::new(elems) }
            .map_err(|_| CoreError::ShapeError(ErrorContext::new("Failed to create slice info")))?;

        // Create a slice that references the original memory-mapped array
        // This is an identity slice for now
        let source = MemoryMappedArray::new::<ndarray::OwnedRepr<A>, E>(
            None,
            &self.file_path,
            self.mode,
            self.offset,
        )?;

        Ok(MemoryMappedSlice::new(source, slice_info))
    }

    fn slice_1d(
        &self,
        range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, ndarray::Ix1>> {
        // Convert to explicit range
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.shape[0],
        };

        if start >= end || end > self.shape[0] {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Invalid slice range {}..{} for array of shape {:?}",
                start, end, self.shape
            ))));
        }

        // Create SliceInfo for 1D array
        let slice_info = unsafe {
            SliceInfo::<Vec<SliceInfoElem>, ndarray::Ix1, ndarray::Ix1>::new(vec![
                SliceInfoElem::Slice {
                    start: start as isize,
                    end: Some(end as isize),
                    step: 1,
                },
            ])
            .map_err(|e| {
                CoreError::ShapeError(ErrorContext::new(format!(
                    "Failed to create slice info: {}",
                    e
                )))
            })?
        };

        // Create a new reference to the same memory-mapped file
        let source = self.clone_ref()?;
        Ok(MemoryMappedSlice::new(source, slice_info))
    }

    fn slice_2d(
        &self,
        row_range: impl RangeBounds<usize>,
        col_range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, ndarray::Ix2>> {
        // Ensure we're working with a 2D array
        if self.shape.len() != 2 {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Expected 2D array, got {}D",
                self.shape.len()
            ))));
        }

        // Convert row range to explicit range
        let row_start = match row_range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let row_end = match row_range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.shape[0],
        };

        // Convert column range to explicit range
        let col_start = match col_range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let col_end = match col_range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.shape[1],
        };

        // Validate ranges
        if row_start >= row_end || row_end > self.shape[0] {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Invalid row slice range {}..{} for array of shape {:?}",
                row_start, row_end, self.shape
            ))));
        }

        if col_start >= col_end || col_end > self.shape[1] {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Invalid column slice range {}..{} for array of shape {:?}",
                col_start, col_end, self.shape
            ))));
        }

        // Create SliceInfo for 2D array
        let slice_info = unsafe {
            SliceInfo::<Vec<SliceInfoElem>, ndarray::Ix2, ndarray::Ix2>::new(vec![
                SliceInfoElem::Slice {
                    start: row_start as isize,
                    end: Some(row_end as isize),
                    step: 1,
                },
                SliceInfoElem::Slice {
                    start: col_start as isize,
                    end: Some(col_end as isize),
                    step: 1,
                },
            ])
            .map_err(|e| {
                CoreError::ShapeError(ErrorContext::new(format!(
                    "Failed to create slice info: {}",
                    e
                )))
            })?
        };

        // Create a new reference to the same memory-mapped file
        let source = self.clone_ref()?;
        Ok(MemoryMappedSlice::new(source, slice_info))
    }
}

#[cfg(test)]
mod tests {
    use super::super::create_mmap;
    use super::*;
    use ndarray::Array2;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_memory_mapped_slice_1d() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_slice_1d.bin");

        // Create a test array and save it to a file
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).unwrap();
        for val in &data {
            file.write_all(&val.to_ne_bytes()).unwrap();
        }
        drop(file);

        // Create a memory-mapped array
        let mmap = MemoryMappedArray::<f64>::open(&file_path, &[100]).unwrap();

        // Create a slice
        let slice = mmap.slice_1d(10..20).unwrap();

        // Load the slice data
        let array = slice.load().unwrap();

        // Check that the slice contains the expected data
        assert_eq!(array.len(), 10);
        for (i, &val) in array.iter().enumerate() {
            assert_eq!(val, (i + 10) as f64);
        }
    }

    #[test]
    fn test_memory_mapped_slice_2d() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_slice_2d.bin");

        // Create a test 2D array and save it to a file using the proper method
        let data = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);

        // Use save_array which handles headers correctly
        MemoryMappedArray::<f64>::save_array(&data, &file_path, None).unwrap();

        // Open using open_zero_copy which handles headers correctly
        let mmap =
            MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwrap();

        // Debug: print shape info
        println!("mmap.shape: {:?}", mmap.shape);
        println!("mmap.size: {}", mmap.size);

        // Debug: print original array to verify
        let orig_array = mmap.as_array::<ndarray::Ix2>().unwrap();
        println!("Original array (first 5x10):");
        for i in 0..5 {
            print!("Row {}: ", i);
            for j in 0..10 {
                print!("{:4.0} ", orig_array[[i, j]]);
            }
            print!("   Expected: ");
            for j in 0..10 {
                print!("{:4} ", i * 10 + j);
            }
            println!();
        }

        // Create a slice
        let slice = mmap.slice_2d(2..5, 3..7).unwrap();

        // Debug: print slice info
        println!("slice.source.shape: {:?}", slice.source.shape);
        println!("slice_info: {:?}", slice.slice_info.as_ref());

        // Load the slice data
        let array = slice.load().unwrap();

        // Check that the slice contains the expected data
        assert_eq!(array.shape(), &[3, 4]);

        // Debug: print the slice content
        println!("Slice content:");
        for i in 0..3 {
            for j in 0..4 {
                print!("{:6.1} ", array[[i, j]]);
            }
            println!();
        }

        for i in 0..3 {
            for j in 0..4 {
                let expected = ((i + 2) * 10 + (j + 3)) as f64;
                let actual = array[[i, j]];
                if actual != expected {
                    println!(
                        "Mismatch at [{}, {}]: expected {}, got {}",
                        i, j, expected, actual
                    );
                }
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    #[ignore = "slice() method implementation needs to be completed"]
    fn test_memory_mapped_slice_with_ndarray_slice_syntax() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_slice_syntax.bin");

        // Create a test 2D array and save it to a file using the proper method
        let data = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);

        // Create a memory-mapped array with proper header
        let mmap = create_mmap::<f64, _, _>(&data, &file_path, AccessMode::Write, 0).unwrap();

        // Create a slice using ndarray's s![] macro
        use ndarray::s;
        let slice = mmap.slice(s![2..5, 3..7]).unwrap();

        // Load the slice data
        let array: ndarray::Array2<f64> = slice.load().unwrap();

        // Check that the slice contains the expected data
        assert_eq!(array.shape(), &[3usize, 4usize]);
        for i in 0..3usize {
            for j in 0..4usize {
                assert_eq!(array[[i, j]], ((i + 2) * 10 + (j + 3)) as f64);
            }
        }
    }
}
