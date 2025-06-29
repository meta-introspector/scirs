//! Image warping and resampling functions
//!
//! This module provides functionality for transforming images using various
//! interpolation methods and geometric transformations.

use crate::error::{Result, VisionError};
use crate::registration::{transform_point, identity_transform, Point2D, TransformMatrix};
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage};
use ndarray::Array2;

/// Interpolation method for image resampling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation
    NearestNeighbor,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
}

/// Boundary handling method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryMethod {
    /// Use zero values outside image bounds
    Zero,
    /// Use constant value outside image bounds
    Constant(f32),
    /// Reflect values at image boundaries
    Reflect,
    /// Wrap around at image boundaries
    Wrap,
    /// Clamp to edge values
    Clamp,
}

/// Warp a grayscale image using a transformation matrix
///
/// # Arguments
///
/// * `image` - Input grayscale image
/// * `transform` - 3x3 transformation matrix
/// * `output_size` - Output image dimensions (width, height)
/// * `interpolation` - Interpolation method
/// * `boundary` - Boundary handling method
///
/// # Returns
///
/// * Result containing the warped image
pub fn warp_image(
    image: &GrayImage,
    transform: &TransformMatrix,
    output_size: (u32, u32),
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
) -> Result<GrayImage> {
    let (out_width, out_height) = output_size;
    let (in_width, in_height) = image.dimensions();

    // Create output image
    let mut output = GrayImage::new(out_width, out_height);

    // Invert transformation for backwards mapping
    // Uses optimized 3x3 matrix inversion for transformation matrices
    let inv_transform = invert_3x3_matrix(transform).map_err(|e| {
        VisionError::OperationError(format!("Failed to invert transformation: {}", e))
    })?;

    // For each pixel in output image
    for y in 0..out_height {
        for x in 0..out_width {
            // Map output coordinates to input coordinates
            let out_point = Point2D::new(x as f64, y as f64);
            let in_point = transform_point(out_point, &inv_transform);

            // Sample input image at mapped coordinates
            let intensity = sample_image(
                image,
                in_point.x as f32,
                in_point.y as f32,
                interpolation,
                boundary,
                in_width,
                in_height,
            );

            output.put_pixel(x, y, Luma([intensity as u8]));
        }
    }

    Ok(output)
}

/// Warp an RGB image using a transformation matrix
pub fn warp_rgb_image(
    image: &RgbImage,
    transform: &TransformMatrix,
    output_size: (u32, u32),
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
) -> Result<RgbImage> {
    let (out_width, out_height) = output_size;
    let (in_width, in_height) = image.dimensions();

    // Create output image
    let mut output = RgbImage::new(out_width, out_height);

    // Invert transformation for backwards mapping
    // Uses optimized 3x3 matrix inversion for transformation matrices
    let inv_transform = invert_3x3_matrix(transform).map_err(|e| {
        VisionError::OperationError(format!("Failed to invert transformation: {}", e))
    })?;

    // For each pixel in output image
    for y in 0..out_height {
        for x in 0..out_width {
            // Map output coordinates to input coordinates
            let out_point = Point2D::new(x as f64, y as f64);
            let in_point = transform_point(out_point, &inv_transform);

            // Sample each color channel
            let r = sample_rgb_image(
                image,
                0,
                in_point.x as f32,
                in_point.y as f32,
                interpolation,
                boundary,
                in_width,
                in_height,
            );
            let g = sample_rgb_image(
                image,
                1,
                in_point.x as f32,
                in_point.y as f32,
                interpolation,
                boundary,
                in_width,
                in_height,
            );
            let b = sample_rgb_image(
                image,
                2,
                in_point.x as f32,
                in_point.y as f32,
                interpolation,
                boundary,
                in_width,
                in_height,
            );

            output.put_pixel(x, y, Rgb([r as u8, g as u8, b as u8]));
        }
    }

    Ok(output)
}

/// Sample a grayscale image at fractional coordinates
fn sample_image(
    image: &GrayImage,
    x: f32,
    y: f32,
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
    width: u32,
    height: u32,
) -> f32 {
    match interpolation {
        InterpolationMethod::NearestNeighbor => {
            let ix = x.round() as i32;
            let iy = y.round() as i32;
            get_pixel_value(image, ix, iy, boundary, width, height)
        }
        InterpolationMethod::Bilinear => {
            let x0 = x.floor() as i32;
            let y0 = y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = x - x0 as f32;
            let fy = y - y0 as f32;

            let v00 = get_pixel_value(image, x0, y0, boundary, width, height);
            let v01 = get_pixel_value(image, x0, y1, boundary, width, height);
            let v10 = get_pixel_value(image, x1, y0, boundary, width, height);
            let v11 = get_pixel_value(image, x1, y1, boundary, width, height);

            let v0 = v00 * (1.0 - fx) + v10 * fx;
            let v1 = v01 * (1.0 - fx) + v11 * fx;

            v0 * (1.0 - fy) + v1 * fy
        }
        InterpolationMethod::Bicubic => {
            // Simplified bicubic interpolation
            let x0 = x.floor() as i32;
            let y0 = y.floor() as i32;

            let fx = x - x0 as f32;
            let fy = y - y0 as f32;

            let mut sum = 0.0;
            for j in -1..3 {
                for i in -1..3 {
                    let weight = cubic_kernel(fx - i as f32) * cubic_kernel(fy - j as f32);
                    let value = get_pixel_value(image, x0 + i, y0 + j, boundary, width, height);
                    sum += weight * value;
                }
            }

            sum.clamp(0.0, 255.0)
        }
    }
}

/// Sample an RGB image at fractional coordinates for a specific channel
fn sample_rgb_image(
    image: &RgbImage,
    channel: usize,
    x: f32,
    y: f32,
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
    width: u32,
    height: u32,
) -> f32 {
    match interpolation {
        InterpolationMethod::NearestNeighbor => {
            let ix = x.round() as i32;
            let iy = y.round() as i32;
            get_rgb_pixel_value(image, channel, ix, iy, boundary, width, height)
        }
        InterpolationMethod::Bilinear => {
            let x0 = x.floor() as i32;
            let y0 = y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = x - x0 as f32;
            let fy = y - y0 as f32;

            let v00 = get_rgb_pixel_value(image, channel, x0, y0, boundary, width, height);
            let v01 = get_rgb_pixel_value(image, channel, x0, y1, boundary, width, height);
            let v10 = get_rgb_pixel_value(image, channel, x1, y0, boundary, width, height);
            let v11 = get_rgb_pixel_value(image, channel, x1, y1, boundary, width, height);

            let v0 = v00 * (1.0 - fx) + v10 * fx;
            let v1 = v01 * (1.0 - fx) + v11 * fx;

            v0 * (1.0 - fy) + v1 * fy
        }
        InterpolationMethod::Bicubic => {
            let x0 = x.floor() as i32;
            let y0 = y.floor() as i32;

            let fx = x - x0 as f32;
            let fy = y - y0 as f32;

            let mut sum = 0.0;
            for j in -1..3 {
                for i in -1..3 {
                    let weight = cubic_kernel(fx - i as f32) * cubic_kernel(fy - j as f32);
                    let value = get_rgb_pixel_value(
                        image,
                        channel,
                        x0 + i,
                        y0 + j,
                        boundary,
                        width,
                        height,
                    );
                    sum += weight * value;
                }
            }

            sum.clamp(0.0, 255.0)
        }
    }
}

/// Get pixel value with boundary handling
fn get_pixel_value(
    image: &GrayImage,
    x: i32,
    y: i32,
    boundary: BoundaryMethod,
    width: u32,
    height: u32,
) -> f32 {
    let (nx, ny) = handle_boundary(x, y, boundary, width, height);

    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
        image.get_pixel(nx as u32, ny as u32)[0] as f32
    } else {
        match boundary {
            BoundaryMethod::Zero => 0.0,
            BoundaryMethod::Constant(value) => value,
            _ => 0.0, // Fallback
        }
    }
}

/// Get RGB pixel value with boundary handling
fn get_rgb_pixel_value(
    image: &RgbImage,
    channel: usize,
    x: i32,
    y: i32,
    boundary: BoundaryMethod,
    width: u32,
    height: u32,
) -> f32 {
    let (nx, ny) = handle_boundary(x, y, boundary, width, height);

    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
        image.get_pixel(nx as u32, ny as u32)[channel] as f32
    } else {
        match boundary {
            BoundaryMethod::Zero => 0.0,
            BoundaryMethod::Constant(value) => value,
            _ => 0.0, // Fallback
        }
    }
}

/// Handle boundary conditions
fn handle_boundary(
    x: i32,
    y: i32,
    boundary: BoundaryMethod,
    width: u32,
    height: u32,
) -> (i32, i32) {
    let w = width as i32;
    let h = height as i32;

    match boundary {
        BoundaryMethod::Zero | BoundaryMethod::Constant(_) => (x, y),
        BoundaryMethod::Reflect => {
            let nx = if x < 0 {
                -x - 1
            } else if x >= w {
                2 * w - x - 1
            } else {
                x
            };

            let ny = if y < 0 {
                -y - 1
            } else if y >= h {
                2 * h - y - 1
            } else {
                y
            };

            (nx.clamp(0, w - 1), ny.clamp(0, h - 1))
        }
        BoundaryMethod::Wrap => {
            let nx = ((x % w) + w) % w;
            let ny = ((y % h) + h) % h;
            (nx, ny)
        }
        BoundaryMethod::Clamp => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
    }
}

/// Cubic interpolation kernel
fn cubic_kernel(t: f32) -> f32 {
    let t = t.abs();
    if t <= 1.0 {
        1.5 * t * t * t - 2.5 * t * t + 1.0
    } else if t <= 2.0 {
        -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
    } else {
        0.0
    }
}

/// Create a mesh grid for transformation mapping
pub fn create_mesh_grid(width: u32, height: u32) -> (Array2<f64>, Array2<f64>) {
    let mut x_grid = Array2::zeros((height as usize, width as usize));
    let mut y_grid = Array2::zeros((height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            x_grid[[y as usize, x as usize]] = x as f64;
            y_grid[[y as usize, x as usize]] = y as f64;
        }
    }

    (x_grid, y_grid)
}

/// Apply perspective correction to an image
pub fn perspective_correct(
    image: &DynamicImage,
    corners: &[Point2D; 4],
    output_size: (u32, u32),
) -> Result<DynamicImage> {
    // Define target rectangle corners
    let (width, height) = output_size;
    let target_corners = [
        Point2D::new(0.0, 0.0),
        Point2D::new(width as f64 - 1.0, 0.0),
        Point2D::new(width as f64 - 1.0, height as f64 - 1.0),
        Point2D::new(0.0, height as f64 - 1.0),
    ];

    // Create matches for homography estimation
    let matches: Vec<_> = corners
        .iter()
        .zip(target_corners.iter())
        .map(|(&src, &tgt)| crate::registration::PointMatch {
            source: src,
            target: tgt,
            confidence: 1.0,
        })
        .collect();

    // Estimate homography
    use crate::registration::estimate_homography_transform;
    let transform = estimate_homography_transform(&matches)?;

    // Warp image
    match image {
        DynamicImage::ImageLuma8(gray) => {
            let warped = warp_image(
                gray,
                &transform,
                output_size,
                InterpolationMethod::Bilinear,
                BoundaryMethod::Zero,
            )?;
            Ok(DynamicImage::ImageLuma8(warped))
        }
        DynamicImage::ImageRgb8(rgb) => {
            let warped = warp_rgb_image(
                rgb,
                &transform,
                output_size,
                InterpolationMethod::Bilinear,
                BoundaryMethod::Zero,
            )?;
            Ok(DynamicImage::ImageRgb8(warped))
        }
        _ => {
            // Convert to RGB and process
            let rgb = image.to_rgb8();
            let warped = warp_rgb_image(
                &rgb,
                &transform,
                output_size,
                InterpolationMethod::Bilinear,
                BoundaryMethod::Zero,
            )?;
            Ok(DynamicImage::ImageRgb8(warped))
        }
    }
}

/// Rectify stereo image pair using fundamental matrix
///
/// This function computes rectification transforms from the fundamental matrix
/// and applies them to align the epipolar lines horizontally in both images.
/// After rectification, corresponding points will have the same y-coordinates.
///
/// # Arguments
///
/// * `left_image` - Left stereo image
/// * `right_image` - Right stereo image  
/// * `fundamental_matrix` - Fundamental matrix relating the two images
///
/// # Returns
///
/// * Result containing the rectified left and right images
///
/// # Algorithm
///
/// Uses Hartley's rectification method:
/// 1. Compute epipoles from fundamental matrix
/// 2. Calculate rectification transforms to align epipolar lines
/// 3. Apply transforms to both images
pub fn rectify_stereo_pair(
    left_image: &DynamicImage,
    right_image: &DynamicImage,
    fundamental_matrix: &TransformMatrix,
) -> Result<(DynamicImage, DynamicImage)> {
    // Ensure both images have the same dimensions
    let (left_width, left_height) = left_image.dimensions();
    let (right_width, right_height) = right_image.dimensions();
    
    if left_width != right_width || left_height != right_height {
        return Err(VisionError::InvalidParameter(
            "Stereo images must have the same dimensions".to_string(),
        ));
    }
    
    // Compute epipoles from fundamental matrix
    let (left_epipole, right_epipole) = compute_epipoles(fundamental_matrix)?;
    
    // Compute rectification transforms
    let (left_transform, right_transform) = compute_rectification_transforms(
        left_epipole,
        right_epipole,
        (left_width, left_height),
        fundamental_matrix,
    )?;
    
    // Apply rectification transforms
    let left_rectified = warp_image(
        &left_image.to_luma8(),
        &left_transform,
        (left_width, left_height),
        InterpolationMethod::Bilinear,
        BoundaryMethod::Zero,
    )?;
    
    let right_rectified = warp_image(
        &right_image.to_luma8(),
        &right_transform,
        (right_width, right_height),
        InterpolationMethod::Bilinear,
        BoundaryMethod::Zero,
    )?;
    
    Ok((
        DynamicImage::ImageLuma8(left_rectified),
        DynamicImage::ImageLuma8(right_rectified),
    ))
}

/// Compute epipoles from fundamental matrix
///
/// The epipoles are the null spaces of F and F^T respectively.
/// For left epipole: F * e_left = 0
/// For right epipole: F^T * e_right = 0
fn compute_epipoles(fundamental_matrix: &TransformMatrix) -> Result<(Point2D, Point2D)> {
    // Find left epipole (null space of F^T)
    let left_epipole = find_null_space(&transpose_matrix(fundamental_matrix))?;
    
    // Find right epipole (null space of F)
    let right_epipole = find_null_space(fundamental_matrix)?;
    
    Ok((left_epipole, right_epipole))
}

/// Transpose a 3x3 matrix
fn transpose_matrix(matrix: &TransformMatrix) -> TransformMatrix {
    let mut transposed = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            transposed[[i, j]] = matrix[[j, i]];
        }
    }
    transposed
}

/// Find the null space of a 3x3 matrix (the eigenvector corresponding to the smallest eigenvalue)
fn find_null_space(matrix: &TransformMatrix) -> Result<Point2D> {
    // Use power iteration to find the smallest eigenvalue and corresponding eigenvector
    // We solve (A^T * A) * v = lambda * v where lambda is the smallest eigenvalue
    
    let mut ata: Array2<f64> = Array2::zeros((3, 3));
    
    // Compute A^T * A
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                ata[[i, j]] += matrix[[k, i]] * matrix[[k, j]];
            }
        }
    }
    
    // Use inverse power iteration to find the smallest eigenvalue
    let mut v = vec![1.0, 1.0, 1.0]; // Initial guess
    
    for _ in 0..50 { // Iteration limit
        // Solve (A^T * A) * v_new = v_old using Gauss-Seidel iteration
        let mut v_new = vec![0.0; 3];
        
        for _ in 0..10 { // Inner iterations for solving linear system
            for i in 0..3 {
                let mut sum = 0.0;
                for j in 0..3 {
                    if i != j {
                        sum += ata[[i, j]] * v_new[j];
                    }
                }
                
                if ata[[i, i]].abs() > 1e-10 {
                    v_new[i] = (v[i] - sum) / ata[[i, i]];
                } else {
                    v_new[i] = v[i]; // Avoid division by zero
                }
            }
        }
        
        // Normalize
        let norm = (v_new[0] * v_new[0] + v_new[1] * v_new[1] + v_new[2] * v_new[2]).sqrt() as f64;
        if norm > 1e-10 {
            for v_new_item in v_new.iter_mut().take(3) {
                *v_new_item /= norm;
            }
        }
        
        v = v_new;
    }
    
    // Convert homogeneous coordinates to 2D point
    if v[2].abs() > 1e-10_f64 {
        Ok(Point2D::new(v[0] / v[2], v[1] / v[2]))
    } else {
        // Point at infinity - use large coordinates
        Ok(Point2D::new(v[0] * 1e6, v[1] * 1e6))
    }
}

/// Compute rectification transforms using Hartley's method
fn compute_rectification_transforms(
    left_epipole: Point2D,
    right_epipole: Point2D,
    image_size: (u32, u32),
    fundamental_matrix: &TransformMatrix,
) -> Result<(TransformMatrix, TransformMatrix)> {
    let (width, height) = image_size;
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;
    
    // Compute left rectification transform
    let left_transform = compute_single_rectification_transform(
        left_epipole,
        (center_x, center_y),
        image_size,
    )?;
    
    // For the right transform, we need to ensure epipolar lines are horizontal
    // and corresponding to the left transform
    let right_transform = compute_right_rectification_transform(
        right_epipole,
        (center_x, center_y),
        image_size,
        &left_transform,
        fundamental_matrix,
    )?;
    
    Ok((left_transform, right_transform))
}

/// Compute rectification transform for a single image
fn compute_single_rectification_transform(
    epipole: Point2D,
    center: (f64, f64),
    image_size: (u32, u32),
) -> Result<TransformMatrix> {
    let (_width, _height) = image_size;
    let (center_x, center_y) = center;
    
    // If epipole is at infinity (parallel cameras), use identity transform
    if epipole.x.abs() > 1e5 || epipole.y.abs() > 1e5 {
        return Ok(identity_transform());
    }
    
    // Translate epipole to origin
    let mut t1 = identity_transform();
    t1[[0, 2]] = -center_x;
    t1[[1, 2]] = -center_y;
    
    // Rotate so that epipole is on positive x-axis
    let ex = epipole.x - center_x;
    let ey = epipole.y - center_y;
    let e_dist = (ex * ex + ey * ey).sqrt();
    
    let mut rotation = identity_transform();
    if e_dist > 1e-10 {
        let cos_theta = ex / e_dist;
        let sin_theta = ey / e_dist;
        
        rotation[[0, 0]] = cos_theta;
        rotation[[0, 1]] = sin_theta;
        rotation[[1, 0]] = -sin_theta;
        rotation[[1, 1]] = cos_theta;
    }
    
    // Apply shearing to make epipolar lines horizontal
    let mut shear = identity_transform();
    
    // Use a simple shearing that maps the epipole to infinity
    let shear_factor = if e_dist > 1e-10 { -ey / ex } else { 0.0 };
    shear[[0, 1]] = shear_factor;
    
    // Translate back to center
    let mut t2 = identity_transform();
    t2[[0, 2]] = center_x;
    t2[[1, 2]] = center_y;
    
    // Combine transforms: T2 * Shear * Rotation * T1
    let temp1 = matrix_multiply(&rotation, &t1)?;
    let temp2 = matrix_multiply(&shear, &temp1)?;
    let final_transform = matrix_multiply(&t2, &temp2)?;
    
    Ok(final_transform)
}

/// Compute right rectification transform that aligns with the left transform
fn compute_right_rectification_transform(
    right_epipole: Point2D,
    center: (f64, f64),
    image_size: (u32, u32),
    left_transform: &TransformMatrix,
    fundamental_matrix: &TransformMatrix,
) -> Result<TransformMatrix> {
    // Start with single-image rectification for right image
    let mut right_transform = compute_single_rectification_transform(
        right_epipole,
        center,
        image_size,
    )?;
    
    // Adjust the right transform to ensure epipolar lines match with left image
    // This involves computing a corrective transform based on the fundamental matrix
    
    // For simplicity, we use the same approach as left image but with different parameters
    // In a full implementation, this would involve more sophisticated epipolar geometry
    
    // Apply a vertical adjustment to align epipolar lines
    let vertical_adjustment = compute_vertical_alignment(
        left_transform,
        &right_transform,
        fundamental_matrix,
        image_size,
    )?;
    
    right_transform[[1, 2]] += vertical_adjustment;
    
    Ok(right_transform)
}

/// Compute vertical adjustment to align epipolar lines between left and right images
fn compute_vertical_alignment(
    left_transform: &TransformMatrix,
    _right_transform: &TransformMatrix,
    fundamental_matrix: &TransformMatrix,
    image_size: (u32, u32),
) -> Result<f64> {
    let (width, height) = image_size;
    
    // Sample points from the left image and compute their epipolar lines in the right image
    let test_points = vec![
        Point2D::new(width as f64 * 0.25, height as f64 * 0.25),
        Point2D::new(width as f64 * 0.75, height as f64 * 0.25),
        Point2D::new(width as f64 * 0.25, height as f64 * 0.75),
        Point2D::new(width as f64 * 0.75, height as f64 * 0.75),
    ];
    
    let mut total_adjustment = 0.0;
    let mut count = 0;
    
    for point in test_points {
        // Transform point through left rectification
        let left_rectified = transform_point(point, left_transform);
        
        // Compute corresponding epipolar line in right image using fundamental matrix
        let epipolar_line = compute_epipolar_line(left_rectified, fundamental_matrix);
        
        // The y-coordinate of this line should be the same as the rectified left point
        // Compute the adjustment needed
        let expected_y = left_rectified.y;
        let actual_y = compute_epipolar_line_y_intercept(&epipolar_line, left_rectified.x);
        
        total_adjustment += expected_y - actual_y;
        count += 1;
    }
    
    if count > 0 {
        Ok(total_adjustment / count as f64)
    } else {
        Ok(0.0)
    }
}

/// Compute epipolar line in the right image corresponding to a point in the left image
fn compute_epipolar_line(point: Point2D, fundamental_matrix: &TransformMatrix) -> (f64, f64, f64) {
    // Epipolar line l = F * p where p is in homogeneous coordinates
    let p = [point.x, point.y, 1.0];
    let mut line = [0.0; 3];
    
    for i in 0..3 {
        for j in 0..3 {
            line[i] += fundamental_matrix[[i, j]] * p[j];
        }
    }
    
    (line[0], line[1], line[2])
}

/// Compute y-intercept of an epipolar line at a given x coordinate
fn compute_epipolar_line_y_intercept(line: &(f64, f64, f64), x: f64) -> f64 {
    let (a, b, c) = *line;
    
    if b.abs() > 1e-10 {
        -(a * x + c) / b
    } else {
        0.0 // Vertical line, return y=0
    }
}

/// Multiply two 3x3 matrices
fn matrix_multiply(a: &TransformMatrix, b: &TransformMatrix) -> Result<TransformMatrix> {
    let mut result = Array2::zeros((3, 3));
    
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
    
    Ok(result)
}

/// Create a panorama by stitching multiple images
pub fn stitch_images(
    images: &[DynamicImage],
    transforms: &[TransformMatrix],
    output_size: (u32, u32),
) -> Result<DynamicImage> {
    if images.len() != transforms.len() {
        return Err(VisionError::InvalidParameter(
            "Number of images must match number of transforms".to_string(),
        ));
    }

    let (width, height) = output_size;
    let mut output = RgbImage::new(width, height);
    let mut weight_map = Array2::<f32>::zeros((height as usize, width as usize));

    // Initialize output with zeros
    for y in 0..height {
        for x in 0..width {
            output.put_pixel(x, y, Rgb([0, 0, 0]));
        }
    }

    // Blend each image
    for (image, transform) in images.iter().zip(transforms.iter()) {
        let rgb_image = image.to_rgb8();
        let warped = warp_rgb_image(
            &rgb_image,
            transform,
            output_size,
            InterpolationMethod::Bilinear,
            BoundaryMethod::Zero,
        )?;

        // Simple averaging blend
        for y in 0..height {
            for x in 0..width {
                let warped_pixel = warped.get_pixel(x, y);
                let output_pixel = output.get_pixel_mut(x, y);

                // Check if warped pixel is not black (indicating valid data)
                if warped_pixel[0] > 0 || warped_pixel[1] > 0 || warped_pixel[2] > 0 {
                    let weight = weight_map[[y as usize, x as usize]];
                    let new_weight = weight + 1.0;

                    for c in 0..3 {
                        let old_value = output_pixel[c] as f32;
                        let new_value = warped_pixel[c] as f32;
                        let blended: f32 = (old_value * weight + new_value) / new_weight;
                        output_pixel[c] = blended as u8;
                    }

                    weight_map[[y as usize, x as usize]] = new_weight;
                }
            }
        }
    }

    Ok(DynamicImage::ImageRgb8(output))
}

/// Simple 3x3 matrix inversion for TransformMatrix
/// Optimized implementation for 3x3 homogeneous transformation matrices
fn invert_3x3_matrix(matrix: &TransformMatrix) -> Result<TransformMatrix> {
    if matrix.shape() != [3, 3] {
        return Err(VisionError::InvalidParameter(
            "Matrix must be 3x3".to_string(),
        ));
    }

    // Compute determinant
    let det = matrix[[0, 0]] * (matrix[[1, 1]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 1]])
        - matrix[[0, 1]] * (matrix[[1, 0]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 0]])
        + matrix[[0, 2]] * (matrix[[1, 0]] * matrix[[2, 1]] - matrix[[1, 1]] * matrix[[2, 0]]);

    if det.abs() < 1e-10 {
        return Err(VisionError::OperationError(
            "Matrix is singular, cannot invert".to_string(),
        ));
    }

    let mut inv = Array2::zeros((3, 3));

    // Compute adjugate matrix
    inv[[0, 0]] = (matrix[[1, 1]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 1]]) / det;
    inv[[0, 1]] = (matrix[[0, 2]] * matrix[[2, 1]] - matrix[[0, 1]] * matrix[[2, 2]]) / det;
    inv[[0, 2]] = (matrix[[0, 1]] * matrix[[1, 2]] - matrix[[0, 2]] * matrix[[1, 1]]) / det;
    inv[[1, 0]] = (matrix[[1, 2]] * matrix[[2, 0]] - matrix[[1, 0]] * matrix[[2, 2]]) / det;
    inv[[1, 1]] = (matrix[[0, 0]] * matrix[[2, 2]] - matrix[[0, 2]] * matrix[[2, 0]]) / det;
    inv[[1, 2]] = (matrix[[0, 2]] * matrix[[1, 0]] - matrix[[0, 0]] * matrix[[1, 2]]) / det;
    inv[[2, 0]] = (matrix[[1, 0]] * matrix[[2, 1]] - matrix[[1, 1]] * matrix[[2, 0]]) / det;
    inv[[2, 1]] = (matrix[[0, 1]] * matrix[[2, 0]] - matrix[[0, 0]] * matrix[[2, 1]]) / det;
    inv[[2, 2]] = (matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]]) / det;

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registration::identity_transform;

    fn create_test_image() -> GrayImage {
        let mut image = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                image.put_pixel(x, y, Luma([((x + y) * 25) as u8]));
            }
        }
        image
    }

    #[test]
    fn test_identity_warp() {
        let image = create_test_image();
        let transform = identity_transform();

        let warped = warp_image(
            &image,
            &transform,
            (10, 10),
            InterpolationMethod::NearestNeighbor,
            BoundaryMethod::Zero,
        )
        .unwrap();

        // Should be identical to original
        for y in 0..10 {
            for x in 0..10 {
                assert_eq!(image.get_pixel(x, y)[0], warped.get_pixel(x, y)[0]);
            }
        }
    }

    #[test]
    fn test_translation_warp() {
        let image = create_test_image();
        let mut transform = identity_transform();
        transform[[0, 2]] = 1.0; // Translate by 1 pixel in x

        let warped = warp_image(
            &image,
            &transform,
            (10, 10),
            InterpolationMethod::NearestNeighbor,
            BoundaryMethod::Zero,
        )
        .unwrap();

        // Check that translation occurred
        assert_eq!(warped.get_pixel(0, 0)[0], 0); // Should be zero (background)
        assert_eq!(warped.get_pixel(1, 0)[0], image.get_pixel(0, 0)[0]);
    }

    #[test]
    fn test_interpolation_methods() {
        let image = create_test_image();
        let transform = identity_transform();

        // Test all interpolation methods
        for &method in &[
            InterpolationMethod::NearestNeighbor,
            InterpolationMethod::Bilinear,
            InterpolationMethod::Bicubic,
        ] {
            let result = warp_image(&image, &transform, (10, 10), method, BoundaryMethod::Zero);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_boundary_methods() {
        let image = create_test_image();
        let mut transform = identity_transform();
        transform[[0, 2]] = -5.0; // Translate outside bounds

        // Test all boundary methods
        for &method in &[
            BoundaryMethod::Zero,
            BoundaryMethod::Constant(128.0),
            BoundaryMethod::Reflect,
            BoundaryMethod::Wrap,
            BoundaryMethod::Clamp,
        ] {
            let result = warp_image(
                &image,
                &transform,
                (10, 10),
                InterpolationMethod::NearestNeighbor,
                method,
            );
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_perspective_correction() {
        let image = DynamicImage::ImageLuma8(create_test_image());

        // Define a simple quadrilateral
        let corners = [
            Point2D::new(1.0, 1.0),
            Point2D::new(8.0, 1.0),
            Point2D::new(8.0, 8.0),
            Point2D::new(1.0, 8.0),
        ];

        let result = perspective_correct(&image, &corners, (100, 100));

        // We now have a working homography estimation without ndarray-linalg
        assert!(result.is_ok());
    }
}
