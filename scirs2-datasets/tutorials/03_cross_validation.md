# Cross-Validation Tutorial

This tutorial covers the comprehensive cross-validation and data splitting utilities provided by SciRS2 datasets, essential for robust machine learning model evaluation.

## Overview

SciRS2 provides advanced cross-validation methods:

- **K-Fold Cross-Validation**: Standard k-fold splitting
- **Stratified K-Fold**: Maintains class distribution across folds
- **Time Series Cross-Validation**: Respects temporal order
- **Group-based Cross-Validation**: Keeps related samples together
- **Train/Test Splitting**: Simple random and stratified splitting
- **Custom Validation Strategies**: Flexible validation schemes

## Basic Train/Test Splitting

### Simple Random Split

```rust
use scirs2_datasets::{load_iris, utils::train_test_split};

let iris = load_iris()?;

// 80% train, 20% test split
let (train, test) = train_test_split(&iris, 0.2, Some(42))?;

println!("Original dataset: {} samples", iris.n_samples());
println!("Training set: {} samples", train.n_samples());
println!("Test set: {} samples", test.n_samples());

// Verify the split
assert_eq!(train.n_samples() + test.n_samples(), iris.n_samples());
```

### Stratified Train/Test Split

```rust
use scirs2_datasets::{load_iris, utils::stratified_train_test_split};

let iris = load_iris()?;

// Maintain class distribution in both sets
if let Some(target) = &iris.target {
    let (train, test) = stratified_train_test_split(&iris, 0.2, Some(42))?;
    
    println!("Stratified split completed");
    println!("Training set: {} samples", train.n_samples());
    println!("Test set: {} samples", test.n_samples());
    
    // Check class distribution preservation
    // (Implementation would analyze target distributions)
}
```

## K-Fold Cross-Validation

### Standard K-Fold

```rust
use scirs2_datasets::{load_boston, k_fold_split};

let boston = load_boston()?;
let n_samples = boston.n_samples();

// 5-fold cross-validation
let folds = k_fold_split(n_samples, 5, true, Some(42))?;

println!("Created {} folds for cross-validation", folds.len());

for (fold_idx, (train_indices, test_indices)) in folds.iter().enumerate() {
    println!("Fold {}: {} train, {} test", 
             fold_idx + 1, train_indices.len(), test_indices.len());
}
```

### Using K-Fold for Model Evaluation

```rust
use scirs2_datasets::{load_iris, k_fold_split};
use ndarray::Array1;

let iris = load_iris()?;
let n_samples = iris.n_samples();

// 10-fold cross-validation
let folds = k_fold_split(n_samples, 10, true, Some(42))?;
let mut fold_scores = Vec::new();

for (fold_idx, (train_indices, test_indices)) in folds.iter().enumerate() {
    // Extract training data
    let train_data = iris.data.select(ndarray::Axis(0), train_indices);
    let train_target = if let Some(target) = &iris.target {
        target.select(ndarray::Axis(0), train_indices)
    } else {
        Array1::zeros(train_indices.len())
    };
    
    // Extract test data
    let test_data = iris.data.select(ndarray::Axis(0), test_indices);
    let test_target = if let Some(target) = &iris.target {
        target.select(ndarray::Axis(0), test_indices)
    } else {
        Array1::zeros(test_indices.len())
    };
    
    // Train and evaluate your model here
    // let model = train_model(&train_data, &train_target);
    // let score = evaluate_model(&model, &test_data, &test_target);
    let score = 0.95; // Placeholder score
    
    fold_scores.push(score);
    println!("Fold {}: score = {:.3}", fold_idx + 1, score);
}

let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
let std_score = {
    let variance = fold_scores.iter()
        .map(|&x| (x - mean_score).powi(2))
        .sum::<f64>() / fold_scores.len() as f64;
    variance.sqrt()
};

println!("Cross-validation results:");
println!("  Mean score: {:.3} ± {:.3}", mean_score, std_score);
```

## Stratified K-Fold Cross-Validation

### Basic Stratified K-Fold

```rust
use scirs2_datasets::{load_wine, stratified_k_fold_split};

let wine = load_wine()?;

if let Some(target) = &wine.target {
    // Stratified 5-fold ensures each fold has similar class distribution
    let folds = stratified_k_fold_split(target, 5, true, Some(42))?;
    
    println!("Created {} stratified folds", folds.len());
    
    for (fold_idx, (train_indices, test_indices)) in folds.iter().enumerate() {
        // Calculate class distribution in this fold
        let test_targets: Vec<_> = test_indices.iter()
            .map(|&i| target[i] as i32)
            .collect();
        
        let mut class_counts = std::collections::HashMap::new();
        for &class in &test_targets {
            *class_counts.entry(class).or_insert(0) += 1;
        }
        
        println!("Fold {}: {} train, {} test, classes: {:?}", 
                 fold_idx + 1, train_indices.len(), test_indices.len(), class_counts);
    }
}
```

### Advanced Stratified Cross-Validation

```rust
use scirs2_datasets::{make_classification, stratified_k_fold_split};

// Generate an imbalanced dataset
let dataset = make_classification(1000, 20, 3, 2, 15, Some(42))?;

if let Some(target) = &dataset.target {
    // Check original class distribution
    let mut original_dist = std::collections::HashMap::new();
    for &class in target.iter() {
        *original_dist.entry(class as i32).or_insert(0) += 1;
    }
    println!("Original class distribution: {:?}", original_dist);
    
    // Create stratified folds
    let folds = stratified_k_fold_split(target, 5, true, Some(42))?;
    
    // Verify stratification
    for (fold_idx, (_, test_indices)) in folds.iter().enumerate() {
        let mut fold_dist = std::collections::HashMap::new();
        for &idx in test_indices {
            let class = target[idx] as i32;
            *fold_dist.entry(class).or_insert(0) += 1;
        }
        
        println!("Fold {} class distribution: {:?}", fold_idx + 1, fold_dist);
    }
}
```

## Time Series Cross-Validation

### Time Series Split

```rust
use scirs2_datasets::{make_time_series, time_series_split};

// Generate time series data
let ts_data = make_time_series(200, 3, true, true, 0.1, Some(42))?;

// Time series cross-validation with 5 folds
// Each fold uses only past data for training
let folds = time_series_split(ts_data.n_samples(), 5, 0.2)?; // 20% test size

println!("Time series cross-validation:");
for (fold_idx, (train_indices, test_indices)) in folds.iter().enumerate() {
    let train_start = train_indices.first().unwrap_or(&0);
    let train_end = train_indices.last().unwrap_or(&0);
    let test_start = test_indices.first().unwrap_or(&0);
    let test_end = test_indices.last().unwrap_or(&0);
    
    println!("Fold {}: train[{}:{}], test[{}:{}]", 
             fold_idx + 1, train_start, train_end, test_start, test_end);
}
```

### Rolling Window Cross-Validation

```rust
use scirs2_datasets::{make_time_series, rolling_window_split};

let ts_data = make_time_series(365, 2, true, true, 0.05, Some(42))?; // Daily data for 1 year

// Rolling window: 30 days training, 7 days testing
let folds = rolling_window_split(
    ts_data.n_samples(), 
    30,  // train_window_size
    7,   // test_window_size  
    7    // step_size
)?;

println!("Rolling window cross-validation:");
println!("Number of folds: {}", folds.len());

for (fold_idx, (train_indices, test_indices)) in folds.iter().enumerate().take(5) {
    println!("Fold {}: train days {}-{}, test days {}-{}", 
             fold_idx + 1,
             train_indices.first().unwrap_or(&0),
             train_indices.last().unwrap_or(&0),
             test_indices.first().unwrap_or(&0),
             test_indices.last().unwrap_or(&0));
}
```

## Group-Based Cross-Validation

### Group K-Fold

```rust
use scirs2_datasets::{make_classification, group_k_fold_split};

let dataset = make_classification(500, 10, 3, 2, 8, Some(42))?;

// Create artificial groups (e.g., patient IDs, document authors, etc.)
let groups: Vec<usize> = (0..dataset.n_samples())
    .map(|i| i / 10) // 10 samples per group
    .collect();

let folds = group_k_fold_split(&groups, 5)?;

println!("Group K-fold cross-validation:");
for (fold_idx, (train_indices, test_indices)) in folds.iter().enumerate() {
    let train_groups: std::collections::HashSet<_> = train_indices.iter()
        .map(|&i| groups[i])
        .collect();
    let test_groups: std::collections::HashSet<_> = test_indices.iter()
        .map(|&i| groups[i])
        .collect();
    
    println!("Fold {}: {} train groups, {} test groups", 
             fold_idx + 1, train_groups.len(), test_groups.len());
    
    // Verify no group overlap
    let overlap: Vec<_> = train_groups.intersection(&test_groups).collect();
    assert!(overlap.is_empty(), "Groups should not overlap between train/test");
}
```

## Leave-One-Out Cross-Validation

### LOO-CV for Small Datasets

```rust
use scirs2_datasets::{load_iris, leave_one_out_split};

let iris = load_iris()?;
let n_samples = iris.n_samples();

if n_samples <= 200 { // Only for small datasets
    let folds = leave_one_out_split(n_samples)?;
    
    println!("Leave-One-Out cross-validation:");
    println!("Number of folds: {}", folds.len());
    println!("Each fold: {} train, 1 test", n_samples - 1);
    
    // Process first few folds as example
    for (fold_idx, (train_indices, test_indices)) in folds.iter().enumerate().take(3) {
        println!("Fold {}: test sample {}", fold_idx + 1, test_indices[0]);
    }
}
```

## Advanced Cross-Validation Strategies

### Nested Cross-Validation

```rust
use scirs2_datasets::{make_classification, k_fold_split};

let dataset = make_classification(200, 15, 3, 2, 10, Some(42))?;
let n_samples = dataset.n_samples();

// Outer CV for performance estimation
let outer_folds = k_fold_split(n_samples, 5, true, Some(42))?;

for (outer_idx, (outer_train, outer_test)) in outer_folds.iter().enumerate() {
    println!("Outer fold {}", outer_idx + 1);
    
    // Inner CV for hyperparameter tuning
    let inner_folds = k_fold_split(outer_train.len(), 3, true, Some(42 + outer_idx as u64))?;
    
    let mut inner_scores = Vec::new();
    for (inner_idx, (inner_train_rel, inner_val_rel)) in inner_folds.iter().enumerate() {
        // Convert relative indices to absolute indices
        let inner_train: Vec<_> = inner_train_rel.iter().map(|&i| outer_train[i]).collect();
        let inner_val: Vec<_> = inner_val_rel.iter().map(|&i| outer_train[i]).collect();
        
        // Hyperparameter tuning would happen here
        let inner_score = 0.85; // Placeholder
        inner_scores.push(inner_score);
        
        println!("  Inner fold {}: score = {:.3}", inner_idx + 1, inner_score);
    }
    
    let best_inner_score = inner_scores.iter().fold(0.0, |a, &b| a.max(b));
    println!("  Best inner score: {:.3}", best_inner_score);
}
```

### Custom Cross-Validation

```rust
use scirs2_datasets::{make_classification, Dataset};
use ndarray::Array1;

// Custom cross-validation function
fn custom_split(dataset: &Dataset, test_ratio: f64, min_samples: usize) 
    -> Result<Vec<(Vec<usize>, Vec<usize>)>, Box<dyn std::error::Error>> {
    let n_samples = dataset.n_samples();
    let n_test = ((n_samples as f64 * test_ratio) as usize).max(min_samples);
    let n_train = n_samples - n_test;
    
    if n_train < min_samples {
        return Err("Not enough samples for training".into());
    }
    
    let mut indices: Vec<usize> = (0..n_samples).collect();
    
    // Use a simple deterministic split for this example
    let train_indices = indices[..n_train].to_vec();
    let test_indices = indices[n_train..].to_vec();
    
    Ok(vec![(train_indices, test_indices)])
}

let dataset = make_classification(100, 10, 2, 1, 8, Some(42))?;
let folds = custom_split(&dataset, 0.3, 20)?;

println!("Custom cross-validation:");
for (fold_idx, (train, test)) in folds.iter().enumerate() {
    println!("Fold {}: {} train, {} test", fold_idx + 1, train.len(), test.len());
}
```

## Cross-Validation Best Practices

### Reproducible Cross-Validation

```rust
use scirs2_datasets::{load_breast_cancer, stratified_k_fold_split};

let cancer = load_breast_cancer()?;

if let Some(target) = &cancer.target {
    // Always use a fixed random seed for reproducibility
    let seed = 42;
    let folds1 = stratified_k_fold_split(target, 5, true, Some(seed))?;
    let folds2 = stratified_k_fold_split(target, 5, true, Some(seed))?;
    
    // Verify reproducibility
    for (i, ((train1, test1), (train2, test2))) in folds1.iter().zip(folds2.iter()).enumerate() {
        assert_eq!(train1, train2, "Training sets should be identical for fold {}", i);
        assert_eq!(test1, test2, "Test sets should be identical for fold {}", i);
    }
    
    println!("Cross-validation is reproducible!");
}
```

### Performance Evaluation

```rust
use scirs2_datasets::{make_classification, stratified_k_fold_split};
use std::time::Instant;

let dataset = make_classification(5000, 50, 5, 2, 30, Some(42))?;

if let Some(target) = &dataset.target {
    let start = Instant::now();
    let folds = stratified_k_fold_split(target, 10, true, Some(42))?;
    let duration = start.elapsed();
    
    println!("Cross-validation setup:");
    println!("  Dataset: {} samples, {} features", dataset.n_samples(), dataset.n_features());
    println!("  Folds: {}", folds.len());
    println!("  Setup time: {:.2}ms", duration.as_millis());
    
    // Calculate memory usage estimate
    let memory_per_fold = std::mem::size_of::<usize>() * dataset.n_samples();
    let total_memory = memory_per_fold * folds.len();
    println!("  Memory usage: ~{:.1} KB", total_memory as f64 / 1024.0);
}
```

### Handling Imbalanced Data

```rust
use scirs2_datasets::{generators::ClassificationConfig, stratified_k_fold_split};

// Create heavily imbalanced dataset
let config = ClassificationConfig {
    n_samples: 1000,
    n_features: 20,
    n_classes: 3,
    weights: Some(vec![0.8, 0.15, 0.05]), // Very imbalanced
    ..Default::default()
};

let dataset = config.generate()?;

if let Some(target) = &dataset.target {
    // Check original distribution
    let mut class_counts = std::collections::HashMap::new();
    for &class in target.iter() {
        *class_counts.entry(class as i32).or_insert(0) += 1;
    }
    println!("Original distribution: {:?}", class_counts);
    
    // Stratified CV maintains proportions
    let folds = stratified_k_fold_split(target, 5, true, Some(42))?;
    
    // Verify each fold maintains similar distribution
    for (fold_idx, (_, test_indices)) in folds.iter().enumerate() {
        let mut fold_counts = std::collections::HashMap::new();
        for &idx in test_indices {
            let class = target[idx] as i32;
            *fold_counts.entry(class).or_insert(0) += 1;
        }
        
        println!("Fold {} distribution: {:?}", fold_idx + 1, fold_counts);
    }
}
```

This tutorial covered the comprehensive cross-validation capabilities of SciRS2. These tools are essential for robust model evaluation, hyperparameter tuning, and performance estimation in machine learning applications.