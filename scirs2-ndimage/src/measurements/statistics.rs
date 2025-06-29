//! Statistical measurement functions for labeled arrays

use ndarray::{Array, Array1, Dimension};
use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};

/// Helper function for safe usize conversion
fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Sum the values of an array for each labeled region
///
/// This function computes the sum of pixel values within each labeled region,
/// which is useful for analyzing connected components or regions of interest.
/// Background pixels (label 0) are automatically excluded from calculations.
///
/// # Arguments
///
/// * `input` - Input array containing values to sum
/// * `labels` - Label array defining regions (same shape as input)
/// * `index` - Specific labels to include (if None, includes all non-zero labels)
///
/// # Returns
///
/// * `Result<Array1<T>>` - Array containing sum of values for each region, indexed by label order
///
/// # Examples
///
/// ## Basic usage with 2D arrays
/// ```
/// use ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::sum_labels;
///
/// // Create input image and corresponding labels
/// let image = array![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0]
/// ];
///
/// let labels = array![
///     [1, 1, 2],
///     [1, 2, 2],
///     [3, 3, 3]
/// ];
///
/// let sums = sum_labels(&image, &labels, None).unwrap();
/// // sums[0] = sum of region 1 = 1+2+4 = 7
/// // sums[1] = sum of region 2 = 3+5+6 = 14  
/// // sums[2] = sum of region 3 = 7+8+9 = 24
/// ```
///
/// ## Computing specific region sums
/// ```
/// use ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::sum_labels;
///
/// let values = array![
///     [10.0, 20.0, 30.0],
///     [40.0, 50.0, 60.0]
/// ];
///
/// let regions = array![
///     [1, 2, 3],
///     [1, 2, 3]
/// ];
///
/// // Only compute sums for regions 1 and 3
/// let partial_sums = sum_labels(&values, &regions, Some(&[1, 3])).unwrap();
/// // Returns sums only for the specified regions
/// assert_eq!(partial_sums.len(), 2);
/// ```
///
/// ## Processing segmented image data
/// ```
/// use ndarray::Array2;
/// use scirs2_ndimage::measurements::sum_labels;
///
/// // Simulate segmented image with intensity values
/// let intensity_image = Array2::from_shape_fn((100, 100), |(i, j)| {
///     ((i + j) as f64).sin().abs() * 255.0
/// });
///
/// // Simulate segmentation labels (e.g., from watershed)
/// let segmentation = Array2::from_shape_fn((100, 100), |(i, j)| {
///     ((i / 10) * 10 + (j / 10)) + 1  // Create grid-like segments
/// });
///
/// let total_intensities = sum_labels(&intensity_image, &segmentation, None).unwrap();
/// // Each element contains total intensity for that segment
/// ```
pub fn sum_labels<T, D>(
    input: &Array<T, D>,
    labels: &Array<usize, D>,
    index: Option<&[usize]>,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if labels.shape() != input.shape() {
        return Err(NdimageError::DimensionError(
            "Labels array must have same shape as input array".to_string(),
        ));
    }

    // Find unique labels
    let unique_labels: std::collections::HashSet<usize> = if let Some(idx) = index {
        idx.iter().cloned().collect()
    } else {
        labels.iter().cloned().collect()
    };

    let mut sorted_labels: Vec<usize> = unique_labels.into_iter().collect();
    sorted_labels.sort();

    // Remove label 0 if it exists (typically background)
    if sorted_labels.first() == Some(&0) {
        sorted_labels.remove(0);
    }

    if sorted_labels.is_empty() {
        return Ok(Array1::<T>::zeros(0));
    }

    // Create a mapping from label to index
    let label_to_idx: std::collections::HashMap<usize, usize> = sorted_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    // Initialize sums array
    let mut sums = vec![T::zero(); sorted_labels.len()];

    // Sum values for each label
    for (input_val, label_val) in input.iter().zip(labels.iter()) {
        if let Some(&idx) = label_to_idx.get(label_val) {
            sums[idx] += *input_val;
        }
    }

    Ok(Array1::from_vec(sums))
}

/// Calculate the mean of an array for each labeled region
///
/// # Arguments
///
/// * `input` - Input array
/// * `labels` - Label array
/// * `index` - Labels to include (if None, includes all labels)
///
/// # Returns
///
/// * `Result<Array1<T>>` - Mean of values for each label
pub fn mean_labels<T, D>(
    input: &Array<T, D>,
    labels: &Array<usize, D>,
    index: Option<&[usize]>,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if labels.shape() != input.shape() {
        return Err(NdimageError::DimensionError(
            "Labels array must have same shape as input array".to_string(),
        ));
    }

    // Get sums and counts for each label
    let sums = sum_labels(input, labels, index)?;
    let counts = count_labels(labels, index)?;

    if sums.len() != counts.len() {
        return Err(NdimageError::InvalidInput(
            "Mismatch between sums and counts arrays".into(),
        ));
    }

    // Calculate means (sum / count for each label)
    let means: Vec<T> = sums
        .iter()
        .zip(counts.iter())
        .map(|(&sum, &count)| {
            if count > 0 {
                sum / safe_usize_to_float(count).unwrap_or(T::one())
            } else {
                T::zero()
            }
        })
        .collect();

    Ok(Array1::from_vec(means))
}

/// Calculate the variance of an array for each labeled region
///
/// Computes the sample variance for pixel values within each labeled region.
/// Variance measures the spread of values around the mean, useful for analyzing
/// region homogeneity and texture properties.
///
/// # Arguments
///
/// * `input` - Input array containing values to analyze
/// * `labels` - Label array defining regions (same shape as input)  
/// * `index` - Specific labels to include (if None, includes all non-zero labels)
///
/// # Returns
///
/// * `Result<Array1<T>>` - Array containing variance for each region
///
/// # Examples
///
/// ## Basic variance calculation
/// ```
/// use ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::variance_labels;
///
/// let values = array![
///     [1.0, 2.0, 10.0],
///     [1.5, 2.5, 10.5],
///     [5.0, 5.0, 5.0]
/// ];
///
/// let regions = array![
///     [1, 1, 2],
///     [1, 1, 2],
///     [3, 3, 3]
/// ];
///
/// let variances = variance_labels(&values, &regions, None).unwrap();
/// // Region 1: variance of [1.0, 2.0, 1.5, 2.5]
/// // Region 2: variance of [10.0, 10.5]
/// // Region 3: variance of [5.0, 5.0, 5.0] = 0.0 (no variation)
/// ```
pub fn variance_labels<T, D>(
    input: &Array<T, D>,
    labels: &Array<usize, D>,
    index: Option<&[usize]>,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if labels.shape() != input.shape() {
        return Err(NdimageError::DimensionError(
            "Labels array must have same shape as input array".to_string(),
        ));
    }

    // Find unique labels
    let unique_labels: std::collections::HashSet<usize> = if let Some(idx) = index {
        idx.iter().cloned().collect()
    } else {
        labels.iter().cloned().collect()
    };

    let mut sorted_labels: Vec<usize> = unique_labels.into_iter().collect();
    sorted_labels.sort();

    // Remove label 0 if it exists (typically background)
    if sorted_labels.first() == Some(&0) {
        sorted_labels.remove(0);
    }

    if sorted_labels.is_empty() {
        return Ok(Array1::<T>::zeros(0));
    }

    // Create a mapping from label to index
    let label_to_idx: std::collections::HashMap<usize, usize> = sorted_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    // First pass: calculate means
    let mut sums = vec![T::zero(); sorted_labels.len()];
    let mut counts = vec![0usize; sorted_labels.len()];

    for (input_val, label_val) in input.iter().zip(labels.iter()) {
        if let Some(&idx) = label_to_idx.get(label_val) {
            sums[idx] += *input_val;
            counts[idx] += 1;
        }
    }

    // Calculate means
    let means: Vec<T> = sums
        .iter()
        .zip(&counts)
        .map(|(&sum, &count)| {
            if count > 0 {
                sum / safe_usize_to_float(count).unwrap_or(T::one())
            } else {
                T::zero()
            }
        })
        .collect();

    // Second pass: calculate variances
    let mut variance_sums = vec![T::zero(); sorted_labels.len()];

    for (input_val, label_val) in input.iter().zip(labels.iter()) {
        if let Some(&idx) = label_to_idx.get(label_val) {
            let diff = *input_val - means[idx];
            variance_sums[idx] += diff * diff;
        }
    }

    // Calculate sample variances (divide by n-1)
    let variances: Vec<T> = variance_sums
        .iter()
        .zip(&counts)
        .map(|(&var_sum, &count)| {
            if count > 1 {
                var_sum / safe_usize_to_float(count - 1).unwrap_or(T::one())
            } else {
                T::zero() // Single pixel regions have zero variance
            }
        })
        .collect();

    Ok(Array1::from_vec(variances))
}

/// Count the number of elements in each labeled region
///
/// # Arguments
///
/// * `labels` - Label array
/// * `index` - Labels to include (if None, includes all labels)
///
/// # Returns
///
/// * `Result<Array1<usize>>` - Count of elements for each label
pub fn count_labels<D>(
    labels: &Array<usize, D>,
    index: Option<&[usize]>,
) -> NdimageResult<Array1<usize>>
where
    D: Dimension,
{
    // Validate inputs
    if labels.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Labels array cannot be 0-dimensional".into(),
        ));
    }

    // Find unique labels
    let unique_labels: std::collections::HashSet<usize> = if let Some(idx) = index {
        idx.iter().cloned().collect()
    } else {
        labels.iter().cloned().collect()
    };

    let mut sorted_labels: Vec<usize> = unique_labels.into_iter().collect();
    sorted_labels.sort();

    // Remove label 0 if it exists (typically background)
    if sorted_labels.first() == Some(&0) {
        sorted_labels.remove(0);
    }

    if sorted_labels.is_empty() {
        return Ok(Array1::<usize>::zeros(0));
    }

    // Create a mapping from label to index
    let label_to_idx: std::collections::HashMap<usize, usize> = sorted_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    // Initialize counts array
    let mut counts = vec![0usize; sorted_labels.len()];

    // Count occurrences of each label
    for &label_val in labels.iter() {
        if let Some(&idx) = label_to_idx.get(&label_val) {
            counts[idx] += 1;
        }
    }

    Ok(Array1::from_vec(counts))
}

/// Calculate histogram of labeled array
///
/// # Arguments
///
/// * `input` - Input array
/// * `min` - Minimum value of range
/// * `max` - Maximum value of range
/// * `bins` - Number of bins
/// * `labels` - Label array (if None, uses whole array)
/// * `index` - Labels to include (if None, includes all labels)
///
/// # Returns
///
/// * `Result<(Array1<usize>, Array1<T>)>` - Histogram counts and bin edges
pub fn histogram<T, D>(
    input: &Array<T, D>,
    min: T,
    max: T,
    bins: usize,
    labels: Option<&Array<usize, D>>,
    _index: Option<&[usize]>,
) -> NdimageResult<(Array1<usize>, Array1<T>)>
where
    T: Float + FromPrimitive + Debug + NumAssign,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if min >= max {
        return Err(NdimageError::InvalidInput(format!(
            "min must be less than max (got min={:?}, max={:?})",
            min, max
        )));
    }

    if bins == 0 {
        return Err(NdimageError::InvalidInput(
            "bins must be greater than 0".into(),
        ));
    }

    if let Some(lab) = labels {
        if lab.shape() != input.shape() {
            return Err(NdimageError::DimensionError(
                "Labels array must have same shape as input array".to_string(),
            ));
        }
    }

    // Placeholder implementation
    let hist = Array1::<usize>::zeros(bins);
    let edges = Array1::<T>::linspace(min, max, bins + 1);

    Ok((hist, edges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2, Array3};

    #[test]
    fn test_sum_labels_basic() {
        let input: Array2<f64> = Array2::eye(3);
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);
        let result =
            sum_labels(&input, &labels, None).expect("sum_labels should succeed for basic test");

        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 3.0, epsilon = 1e-10); // Sum of diagonal elements
    }

    #[test]
    fn test_sum_labels_multiple_regions() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let labels = array![[1, 1, 2], [1, 2, 2], [3, 3, 3]];

        let sums = sum_labels(&input, &labels, None)
            .expect("sum_labels should succeed for multiple regions test");
        assert_eq!(sums.len(), 3);
        assert_abs_diff_eq!(sums[0], 1.0 + 2.0 + 4.0, epsilon = 1e-10); // Region 1
        assert_abs_diff_eq!(sums[1], 3.0 + 5.0 + 6.0, epsilon = 1e-10); // Region 2
        assert_abs_diff_eq!(sums[2], 7.0 + 8.0 + 9.0, epsilon = 1e-10); // Region 3
    }

    #[test]
    fn test_sum_labels_with_background() {
        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array![[0, 1], [1, 2]]; // Label 0 is background

        let sums = sum_labels(&input, &labels, None)
            .expect("sum_labels should succeed with background test");
        assert_eq!(sums.len(), 2); // Should exclude background (label 0)
        assert_abs_diff_eq!(sums[0], 2.0 + 3.0, epsilon = 1e-10); // Region 1
        assert_abs_diff_eq!(sums[1], 4.0, epsilon = 1e-10); // Region 2
    }

    #[test]
    fn test_sum_labels_selective_index() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let labels = array![[1, 2, 3], [1, 2, 3]];

        let sums = sum_labels(&input, &labels, Some(&[1, 3]))
            .expect("sum_labels should succeed with selective index test");
        assert_eq!(sums.len(), 2); // Only regions 1 and 3
        assert_abs_diff_eq!(sums[0], 1.0 + 4.0, epsilon = 1e-10); // Region 1
        assert_abs_diff_eq!(sums[1], 3.0 + 6.0, epsilon = 1e-10); // Region 3
    }

    #[test]
    fn test_sum_labels_edge_cases() {
        // Empty result case
        let input = array![[1.0, 2.0]];
        let labels = array![[0, 0]]; // All background
        let sums = sum_labels(&input, &labels, None)
            .expect("sum_labels should succeed for empty result test");
        assert_eq!(sums.len(), 0);

        // Single pixel regions
        let input2 = array![[1.0, 2.0, 3.0]];
        let labels2 = array![[1, 2, 3]];
        let sums2 = sum_labels(&input2, &labels2, None)
            .expect("sum_labels should succeed for single pixel test");
        assert_eq!(sums2.len(), 3);
        assert_abs_diff_eq!(sums2[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sums2[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sums2[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sum_labels_3d() {
        let input = Array3::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k) as f64);
        let labels = Array3::from_shape_fn((2, 2, 2), |(i, j, _k)| if i == j { 1 } else { 2 });

        let sums =
            sum_labels(&input, &labels, None).expect("sum_labels should succeed for 3D test");
        assert_eq!(sums.len(), 2);
        assert!(sums[0] > 0.0);
        assert!(sums[1] > 0.0);
    }

    #[test]
    fn test_mean_labels_basic() {
        let input: Array2<f64> = Array2::eye(3);
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);
        let result =
            mean_labels(&input, &labels, None).expect("mean_labels should succeed for basic test");

        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 3.0 / 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_labels_multiple_regions() {
        let input = array![[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]];
        let labels = array![[1, 1, 2], [1, 2, 2]];

        let means = mean_labels(&input, &labels, None)
            .expect("mean_labels should succeed for multiple regions test");
        assert_eq!(means.len(), 2);
        assert_abs_diff_eq!(means[0], (2.0 + 4.0 + 8.0) / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(means[1], (6.0 + 10.0 + 12.0) / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variance_labels_basic() {
        let input = array![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
        let labels = array![[1, 1, 2], [1, 2, 2]];

        let variances = variance_labels(&input, &labels, None)
            .expect("variance_labels should succeed for basic test");
        assert_eq!(variances.len(), 2);

        // Manual variance calculation for region 1: [1, 3, 2]
        // Mean = 2.0, variance = ((1-2)² + (3-2)² + (2-2)²) / (3-1) = (1+1+0)/2 = 1.0
        assert_abs_diff_eq!(variances[0], 1.0, epsilon = 1e-10);

        // Manual variance calculation for region 2: [5, 4, 6]
        // Mean = 5.0, variance = ((5-5)² + (4-5)² + (6-5)²) / (3-1) = (0+1+1)/2 = 1.0
        assert_abs_diff_eq!(variances[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variance_labels_zero_variance() {
        let input = array![[5.0, 5.0, 3.0], [5.0, 3.0, 3.0]];
        let labels = array![[1, 1, 2], [1, 2, 2]];

        let variances = variance_labels(&input, &labels, None)
            .expect("variance_labels should succeed for zero variance test");
        assert_eq!(variances.len(), 2);
        assert_abs_diff_eq!(variances[0], 0.0, epsilon = 1e-10); // All values are 5.0
        assert_abs_diff_eq!(variances[1], 0.0, epsilon = 1e-10); // All values are 3.0
    }

    #[test]
    fn test_variance_labels_single_pixel() {
        let input = array![[1.0, 2.0, 3.0]];
        let labels = array![[1, 2, 3]]; // Each pixel is its own region

        let variances = variance_labels(&input, &labels, None)
            .expect("variance_labels should succeed for single pixel test");
        assert_eq!(variances.len(), 3);
        // Single pixel regions should have zero variance
        assert_abs_diff_eq!(variances[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(variances[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(variances[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_count_labels_basic() {
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);
        let result =
            count_labels(&labels, None).expect("count_labels should succeed for basic test");

        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 9); // 3x3 grid with all label 1
    }

    #[test]
    fn test_count_labels_multiple_regions() {
        let labels = array![[1, 1, 2, 2], [1, 3, 3, 2], [4, 4, 4, 4]];

        let counts = count_labels(&labels, None)
            .expect("count_labels should succeed for multiple regions test");
        assert_eq!(counts.len(), 4);
        assert_eq!(counts[0], 3); // Region 1: 3 pixels
        assert_eq!(counts[1], 3); // Region 2: 3 pixels
        assert_eq!(counts[2], 2); // Region 3: 2 pixels
        assert_eq!(counts[3], 4); // Region 4: 4 pixels
    }

    #[test]
    fn test_count_labels_with_background() {
        let labels = array![[0, 1, 1], [0, 2, 2], [0, 0, 3]];

        let counts =
            count_labels(&labels, None).expect("count_labels should succeed with background test");
        assert_eq!(counts.len(), 3); // Should exclude background (label 0)
        assert_eq!(counts[0], 2); // Region 1: 2 pixels
        assert_eq!(counts[1], 2); // Region 2: 2 pixels
        assert_eq!(counts[2], 1); // Region 3: 1 pixel
    }

    #[test]
    fn test_error_handling() {
        // Test dimension mismatch
        let input = array![[1.0, 2.0]];
        let labels = array![[1], [2]]; // Wrong shape

        assert!(sum_labels(&input, &labels, None).is_err());
        assert!(mean_labels(&input, &labels, None).is_err());
        assert!(variance_labels(&input, &labels, None).is_err());

        // Test 0-dimensional input
        let input_0d = ndarray::arr0(1.0);
        let labels_0d = ndarray::arr0(1);

        assert!(sum_labels(&input_0d, &labels_0d, None).is_err());
        assert!(mean_labels(&input_0d, &labels_0d, None).is_err());
        assert!(variance_labels(&input_0d, &labels_0d, None).is_err());
        assert!(count_labels(&labels_0d, None).is_err());
    }

    #[test]
    fn test_high_dimensional_arrays() {
        // Test 4D arrays
        let input = Array::from_shape_fn((2, 2, 2, 2), |(i, j, k, l)| (i + j + k + l) as f64);
        let labels = Array::from_shape_fn((2, 2, 2, 2), |(i, j, _k, _l)| i + j + 1);

        let sums =
            sum_labels(&input, &labels, None).expect("sum_labels should succeed for 4D test");
        let means =
            mean_labels(&input, &labels, None).expect("mean_labels should succeed for 4D test");
        let variances = variance_labels(&input, &labels, None)
            .expect("variance_labels should succeed for 4D test");
        let counts = count_labels(&labels, None).expect("count_labels should succeed for 4D test");

        assert!(!sums.is_empty());
        assert!(!means.is_empty());
        assert!(!variances.is_empty());
        assert!(!counts.is_empty());
        assert_eq!(sums.len(), means.len());
        assert_eq!(means.len(), variances.len());
        assert_eq!(variances.len(), counts.len());
    }
}
