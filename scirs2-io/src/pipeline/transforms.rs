//! Common data transformations for pipelines

use super::*;
use crate::error::Result;
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, DataMut, Dimension};
use num_traits::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Normalization transformer
pub struct NormalizeTransform<T> {
    method: NormalizationMethod,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub enum NormalizationMethod {
    MinMax { min: f64, max: f64 },
    ZScore,
    L1,
    L2,
    MaxAbs,
}

impl<T> NormalizeTransform<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    pub fn new(method: NormalizationMethod) -> Self {
        Self {
            method,
            _phantom: PhantomData,
        }
    }
}

impl<T> DataTransformer for NormalizeTransform<T>
where
    T: Float + FromPrimitive + Send + Sync + 'static,
{
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(array) = data.downcast::<Array2<T>>() {
            let normalized = match &self.method {
                NormalizationMethod::MinMax { min, max } => {
                    normalize_minmax(*array, T::from_f64(*min).unwrap(), T::from_f64(*max).unwrap())
                }
                NormalizationMethod::ZScore => normalize_zscore(*array),
                NormalizationMethod::L1 => normalize_l1(*array),
                NormalizationMethod::L2 => normalize_l2(*array),
                NormalizationMethod::MaxAbs => normalize_maxabs(*array),
            };
            Ok(Box::new(normalized) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other("Invalid data type for normalization".to_string()))
        }
    }
}

fn normalize_minmax<T>(mut array: Array2<T>, new_min: T, new_max: T) -> Array2<T>
where
    T: Float + FromPrimitive,
{
    let min = array.iter().fold(T::infinity(), |a, &b| a.min(b));
    let max = array.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
    let range = max - min;
    
    if range > T::zero() {
        let scale = (new_max - new_min) / range;
        array.mapv_inplace(|x| (x - min) * scale + new_min);
    }
    
    array
}

fn normalize_zscore<T>(mut array: Array2<T>) -> Array2<T>
where
    T: Float + FromPrimitive,
{
    let n = T::from_usize(array.len()).unwrap();
    let mean = array.iter().fold(T::zero(), |a, &b| a + b) / n;
    let variance = array.iter().fold(T::zero(), |a, &b| {
        let diff = b - mean;
        a + diff * diff
    }) / n;
    let std = variance.sqrt();
    
    if std > T::zero() {
        array.mapv_inplace(|x| (x - mean) / std);
    }
    
    array
}

fn normalize_l1<T>(mut array: Array2<T>) -> Array2<T>
where
    T: Float,
{
    for mut row in array.axis_iter_mut(Axis(0)) {
        let norm = row.iter().fold(T::zero(), |a, &b| a + b.abs());
        if norm > T::zero() {
            row.mapv_inplace(|x| x / norm);
        }
    }
    array
}

fn normalize_l2<T>(mut array: Array2<T>) -> Array2<T>
where
    T: Float,
{
    for mut row in array.axis_iter_mut(Axis(0)) {
        let norm = row.iter().fold(T::zero(), |a, &b| a + b * b).sqrt();
        if norm > T::zero() {
            row.mapv_inplace(|x| x / norm);
        }
    }
    array
}

fn normalize_maxabs<T>(mut array: Array2<T>) -> Array2<T>
where
    T: Float,
{
    let max_abs = array.iter().fold(T::zero(), |a, &b| a.max(b.abs()));
    if max_abs > T::zero() {
        array.mapv_inplace(|x| x / max_abs);
    }
    array
}

/// Reshape transformer
pub struct ReshapeTransform {
    new_shape: Vec<usize>,
}

impl ReshapeTransform {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { new_shape: shape }
    }
}

impl DataTransformer for ReshapeTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(array) = data.downcast::<Array2<f64>>() {
            let total_elements: usize = self.new_shape.iter().product();
            if array.len() != total_elements {
                return Err(IoError::Other(format!(
                    "Cannot reshape array of size {} to shape {:?}",
                    array.len(),
                    self.new_shape
                )));
            }
            
            // Convert to 1D, then reshape
            let flat: Vec<f64> = array.into_iter().collect();
            let reshaped = Array2::from_shape_vec(
                (self.new_shape[0], self.new_shape[1]),
                flat
            ).map_err(|e| IoError::Other(e.to_string()))?;
            
            Ok(Box::new(reshaped) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other("Invalid data type for reshape".to_string()))
        }
    }
}

/// Type conversion transformer
pub struct TypeConvertTransform<From, To> {
    _from: PhantomData<From>,
    _to: PhantomData<To>,
}

impl<From, To> TypeConvertTransform<From, To> {
    pub fn new() -> Self {
        Self {
            _from: PhantomData,
            _to: PhantomData,
        }
    }
}

impl<From, To> DataTransformer for TypeConvertTransform<From, To>
where
    From: 'static + Send + Sync,
    To: 'static + Send + Sync + std::convert::From<From>,
{
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(from_data) = data.downcast::<From>() {
            let to_data: To = To::from(*from_data);
            Ok(Box::new(to_data) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other("Type conversion failed".to_string()))
        }
    }
}

/// Aggregation transformer
pub struct AggregateTransform {
    method: AggregationMethod,
    axis: Option<Axis>,
}

#[derive(Debug, Clone)]
pub enum AggregationMethod {
    Sum,
    Mean,
    Min,
    Max,
    Std,
    Var,
}

impl AggregateTransform {
    pub fn new(method: AggregationMethod, axis: Option<Axis>) -> Self {
        Self { method, axis }
    }
}

impl DataTransformer for AggregateTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(array) = data.downcast::<Array2<f64>>() {
            let result = match (&self.method, self.axis) {
                (AggregationMethod::Sum, Some(axis)) => {
                    Box::new(array.sum_axis(axis)) as Box<dyn Any + Send + Sync>
                }
                (AggregationMethod::Mean, Some(axis)) => {
                    Box::new(array.mean_axis(axis).unwrap()) as Box<dyn Any + Send + Sync>
                }
                (AggregationMethod::Sum, None) => {
                    Box::new(array.sum()) as Box<dyn Any + Send + Sync>
                }
                (AggregationMethod::Mean, None) => {
                    Box::new(array.mean().unwrap()) as Box<dyn Any + Send + Sync>
                }
                _ => return Err(IoError::Other("Unsupported aggregation".to_string())),
            };
            Ok(result)
        } else {
            Err(IoError::Other("Invalid data type for aggregation".to_string()))
        }
    }
}

/// Encoding transformer for categorical data
pub struct EncodingTransform {
    method: EncodingMethod,
}

#[derive(Debug, Clone)]
pub enum EncodingMethod {
    OneHot,
    Label,
    Ordinal(Vec<String>),
}

impl EncodingTransform {
    pub fn new(method: EncodingMethod) -> Self {
        Self { method }
    }
}

impl DataTransformer for EncodingTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(categories) = data.downcast::<Vec<String>>() {
            match &self.method {
                EncodingMethod::Label => {
                    let mut label_map = HashMap::new();
                    let mut next_label = 0;
                    
                    let encoded: Vec<i32> = categories
                        .iter()
                        .map(|cat| {
                            *label_map.entry(cat.clone()).or_insert_with(|| {
                                let label = next_label;
                                next_label += 1;
                                label
                            })
                        })
                        .collect();
                    
                    Ok(Box::new(encoded) as Box<dyn Any + Send + Sync>)
                }
                EncodingMethod::OneHot => {
                    let unique_categories: Vec<String> = {
                        let mut cats = categories.clone();
                        cats.sort();
                        cats.dedup();
                        cats
                    };
                    
                    let n_categories = unique_categories.len();
                    let n_samples = categories.len();
                    let mut encoded = Array2::<f64>::zeros((n_samples, n_categories));
                    
                    for (i, cat) in categories.iter().enumerate() {
                        if let Ok(j) = unique_categories.iter().position(|c| c == cat) {
                            encoded[[i, j]] = 1.0;
                        }
                    }
                    
                    Ok(Box::new(encoded) as Box<dyn Any + Send + Sync>)
                }
                EncodingMethod::Ordinal(order) => {
                    let encoded: Result<Vec<i32>> = categories
                        .iter()
                        .map(|cat| {
                            order.iter().position(|o| o == cat)
                                .map(|pos| pos as i32)
                                .ok_or_else(|| IoError::Other(format!("Unknown category: {}", cat)))
                        })
                        .collect();
                    
                    Ok(Box::new(encoded?) as Box<dyn Any + Send + Sync>)
                }
            }
        } else {
            Err(IoError::Other("Invalid data type for encoding".to_string()))
        }
    }
}

/// Missing value imputation transformer
pub struct ImputeTransform {
    strategy: ImputationStrategy,
}

#[derive(Debug, Clone)]
pub enum ImputationStrategy {
    Mean,
    Median,
    Mode,
    Constant(f64),
    Forward,
    Backward,
}

impl ImputeTransform {
    pub fn new(strategy: ImputationStrategy) -> Self {
        Self { strategy }
    }
}

impl DataTransformer for ImputeTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(mut array) = data.downcast::<Array2<Option<f64>>>() {
            match &self.strategy {
                ImputationStrategy::Mean => {
                    for mut col in array.axis_iter_mut(Axis(1)) {
                        let valid_values: Vec<f64> = col.iter()
                            .filter_map(|&x| x)
                            .collect();
                        
                        if !valid_values.is_empty() {
                            let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                            col.mapv_inplace(|x| x.unwrap_or(mean));
                        }
                    }
                }
                ImputationStrategy::Constant(value) => {
                    array.mapv_inplace(|x| x.unwrap_or(*value));
                }
                _ => return Err(IoError::Other("Unsupported imputation strategy".to_string())),
            }
            
            // Convert to Array2<f64> after imputation
            let imputed: Array2<f64> = array.mapv(|x| x.unwrap_or(0.0));
            Ok(Box::new(imputed) as Box<dyn Any + Send + Sync>)
        } else {
            Err(IoError::Other("Invalid data type for imputation".to_string()))
        }
    }
}

/// Outlier detection and removal transformer
pub struct OutlierTransform {
    method: OutlierMethod,
    threshold: f64,
}

#[derive(Debug, Clone)]
pub enum OutlierMethod {
    ZScore,
    IQR,
    IsolationForest,
}

impl OutlierTransform {
    pub fn new(method: OutlierMethod, threshold: f64) -> Self {
        Self { method, threshold }
    }
}

impl DataTransformer for OutlierTransform {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> {
        if let Ok(array) = data.downcast::<Array2<f64>>() {
            match &self.method {
                OutlierMethod::ZScore => {
                    let mean = array.mean().unwrap();
                    let std = array.std(0.0);
                    
                    let filtered: Vec<Vec<f64>> = array.axis_iter(Axis(0))
                        .filter(|row| {
                            row.iter().all(|&x| ((x - mean) / std).abs() <= self.threshold)
                        })
                        .map(|row| row.to_vec())
                        .collect();
                    
                    if filtered.is_empty() {
                        return Err(IoError::Other("All data filtered as outliers".to_string()));
                    }
                    
                    let n_rows = filtered.len();
                    let n_cols = filtered[0].len();
                    let flat: Vec<f64> = filtered.into_iter().flatten().collect();
                    
                    let result = Array2::from_shape_vec((n_rows, n_cols), flat)
                        .map_err(|e| IoError::Other(e.to_string()))?;
                    
                    Ok(Box::new(result) as Box<dyn Any + Send + Sync>)
                }
                _ => Err(IoError::Other("Unsupported outlier method".to_string())),
            }
        } else {
            Err(IoError::Other("Invalid data type for outlier detection".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_normalize_minmax() {
        let transform = NormalizeTransform::<f64>::new(NormalizationMethod::MinMax { min: 0.0, max: 1.0 });
        let data = Box::new(arr2(&[[1.0, 2.0], [3.0, 4.0]])) as Box<dyn Any + Send + Sync>;
        let result = transform.transform(data).unwrap();
        let normalized = result.downcast::<Array2<f64>>().unwrap();
        
        assert!((normalized[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((normalized[[1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_encoding_label() {
        let transform = EncodingTransform::new(EncodingMethod::Label);
        let data = Box::new(vec!["cat".to_string(), "dog".to_string(), "cat".to_string()]) as Box<dyn Any + Send + Sync>;
        let result = transform.transform(data).unwrap();
        let encoded = result.downcast::<Vec<i32>>().unwrap();
        
        assert_eq!(*encoded, vec![0, 1, 0]);
    }
}