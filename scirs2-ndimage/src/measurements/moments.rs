//! Moment calculation functions for arrays

use ndarray::{Array, Dimension};
use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};

/// Helper function for safe conversion from usize to float
fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Find the center of mass (centroid) of an array
///
/// Computes the intensity-weighted centroid of an n-dimensional array.
/// The center of mass is calculated as the average position of all pixels,
/// weighted by their intensity values. This is useful for object localization,
/// tracking, and geometric analysis.
///
/// # Arguments
///
/// * `input` - Input array containing intensity values
///
/// # Returns
///
/// * `Result<Vec<T>>` - Center of mass coordinates, one per dimension
///
/// # Examples
///
/// ## Basic 1D center of mass
/// ```
/// use ndarray::Array1;
/// use scirs2_ndimage::measurements::center_of_mass;
///
/// // Simple 1D signal with peak at position 2
/// let signal = Array1::from_vec(vec![0.0, 1.0, 5.0, 1.0, 0.0]);
/// let centroid = center_of_mass(&signal)?;
///
/// // Center of mass should be close to position 2 (where the peak is)
/// assert!((centroid[0] - 2.0).abs() < 0.1);
/// ```
///
/// ## 2D object localization
/// ```
/// use ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::center_of_mass;
///
/// // Create a 2D object (bright square in upper-left)
/// let mut image = Array2::zeros((10, 10));
/// for i in 2..5 {
///     for j in 2..5 {
///         image[[i, j]] = 10.0;
///     }
/// }
///
/// let centroid = center_of_mass(&image)?;
/// // Centroid should be approximately at (3, 3) - center of the bright square
/// assert!((centroid[0] - 3.0).abs() < 0.1);
/// assert!((centroid[1] - 3.0).abs() < 0.1);
/// ```
///
/// ## Intensity-weighted centroid
/// ```
/// use ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::center_of_mass;
///
/// // Create object with non-uniform intensity distribution
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 2.0, 0.0],
///     [0.0, 3.0, 6.0, 0.0],  // Higher intensities toward bottom-right
///     [0.0, 0.0, 0.0, 0.0]
/// ];
///
/// let centroid = center_of_mass(&image)?;
/// // Centroid will be shifted toward higher intensity pixels
/// // Should be closer to (2, 2) than (1.5, 1.5) due to intensity weighting
/// ```
///
/// ## 3D volume center of mass
/// ```
/// use ndarray::Array3;
/// use scirs2_ndimage::measurements::center_of_mass;
///
/// // Create a 3D volume with a bright cube in one corner
/// let mut volume = Array3::zeros((20, 20, 20));
/// for i in 5..10 {
///     for j in 5..10 {
///         for k in 5..10 {
///             volume[[i, j, k]] = 1.0;
///         }
///     }
/// }
///
/// let centroid = center_of_mass(&volume)?;
/// // Centroid should be at approximately (7.5, 7.5, 7.5)
/// assert_eq!(centroid.len(), 3); // 3D coordinates
/// ```
///
/// ## Binary object analysis
/// ```
/// use ndarray::Array2;
/// use scirs2_ndimage::measurements::center_of_mass;
///
/// // Binary image (0.0 and 1.0 values only)
/// let binary = Array2::from_shape_fn((50, 50), |(i, j)| {
///     if ((i as f64 - 25.0).powi(2) + (j as f64 - 25.0).powi(2)).sqrt() < 10.0 {
///         1.0
///     } else {
///         0.0
///     }
/// });
///
/// let centroid = center_of_mass(&binary)?;
/// // For a circular object centered at (25, 25), centroid should be near center
/// assert!((centroid[0] - 25.0).abs() < 1.0);
/// assert!((centroid[1] - 25.0).abs() < 1.0);
/// ```
///
/// # Special Cases
///
/// - If the total mass (sum of all values) is zero, returns the geometric center of the array
/// - For binary images, equivalent to finding the centroid of the foreground region
/// - Subpixel precision is maintained for accurate localization
pub fn center_of_mass<T, D>(input: &Array<T, D>) -> NdimageResult<Vec<T>>
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

    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    let ndim = input.ndim();
    let shape = input.shape();

    // Calculate total mass (sum of all values)
    let total_mass = input.sum();

    if total_mass == T::zero() {
        // If total mass is zero, return center of array
        let center: Result<Vec<T>, NdimageError> = shape
            .iter()
            .map(|&dim| {
                let dim_t = safe_usize_to_float(dim)?;
                Ok(dim_t / (T::one() + T::one()))
            })
            .collect();
        let center = center?;
        return Ok(center);
    }

    // Calculate center of mass for each dimension
    let mut center_of_mass = vec![T::zero(); ndim];

    // Convert to dynamic array for easier indexing
    let input_dyn = input.clone().into_dyn();

    // Iterate through all elements in the array
    for (idx, &value) in input_dyn.indexed_iter() {
        if value != T::zero() {
            // Add weighted coordinates
            for (dim, &coord) in idx.as_array_view().iter().enumerate() {
                let coord_t = safe_usize_to_float(coord)?;
                center_of_mass[dim] += coord_t * value;
            }
        }
    }

    // Normalize by total mass
    for coord in center_of_mass.iter_mut() {
        *coord /= total_mass;
    }

    Ok(center_of_mass)
}

/// Find the moment of inertia tensor of an array
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// * `Result<Array<T, ndarray::Ix2>>` - Moment of inertia tensor
pub fn moments_inertia_tensor<T, D>(input: &Array<T, D>) -> NdimageResult<Array<T, ndarray::Ix2>>
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

    // Placeholder implementation
    let dim = input.ndim();
    Ok(Array::<T, _>::zeros((dim, dim)))
}

/// Calculate image moments
///
/// # Arguments
///
/// * `input` - Input array
/// * `order` - Maximum order of moments to calculate
///
/// # Returns
///
/// * `Result<Array<T, ndarray::Ix1>>` - Array of moments
pub fn moments<T, D>(input: &Array<T, D>, order: usize) -> NdimageResult<Array<T, ndarray::Ix1>>
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

    // Placeholder implementation
    let num_moments = (order + 1).pow(input.ndim() as u32);
    Ok(Array::<T, _>::zeros(num_moments))
}

/// Calculate central moments (moments around centroid)
///
/// # Arguments
///
/// * `input` - Input array
/// * `order` - Maximum order of moments to calculate
/// * `center` - Center coordinates (if None, uses center of mass)
///
/// # Returns
///
/// * `Result<Array<T, ndarray::Ix1>>` - Array of central moments
pub fn central_moments<T, D>(
    input: &Array<T, D>,
    order: usize,
    center: Option<&[T]>,
) -> NdimageResult<Array<T, ndarray::Ix1>>
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

    if let Some(c) = center {
        if c.len() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Center must have same length as input dimensions (got {} expected {})",
                c.len(),
                input.ndim()
            )));
        }
    }

    // Placeholder implementation
    let num_moments = (order + 1).pow(input.ndim() as u32);
    Ok(Array::<T, _>::zeros(num_moments))
}

/// Calculate normalized moments
///
/// # Arguments
///
/// * `input` - Input array
/// * `order` - Maximum order of moments to calculate
///
/// # Returns
///
/// * `Result<Array<T, ndarray::Ix1>>` - Array of normalized moments
pub fn normalized_moments<T, D>(
    input: &Array<T, D>,
    order: usize,
) -> NdimageResult<Array<T, ndarray::Ix1>>
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

    // Placeholder implementation
    let num_moments = (order + 1).pow(input.ndim() as u32);
    Ok(Array::<T, _>::zeros(num_moments))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_center_of_mass() {
        let input: Array2<f64> = Array2::eye(3);
        let com = center_of_mass(&input)
            .expect("center_of_mass should succeed for test");
        assert_eq!(com.len(), input.ndim());
    }

    #[test]
    fn test_moments_inertia_tensor() {
        let input: Array2<f64> = Array2::eye(3);
        let tensor = moments_inertia_tensor(&input)
            .expect("moments_inertia_tensor should succeed for test");
        assert_eq!(tensor.shape(), &[input.ndim(), input.ndim()]);
    }

    #[test]
    fn test_moments() {
        let input: Array2<f64> = Array2::eye(3);
        let order = 2;
        let mom = moments(&input, order)
            .expect("moments should succeed for test");
        assert!(!mom.is_empty());
    }
}
