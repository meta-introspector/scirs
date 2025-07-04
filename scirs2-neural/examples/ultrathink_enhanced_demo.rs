//! Enhanced neural network demonstration showcasing ultrathink mode improvements
//!
//! This example demonstrates the expanded functionality in scirs2-neural including:
//! - Dense, Conv2D, and LSTM layers
//! - Multiple activation functions
//! - Loss functions (MSE, CrossEntropy, Binary CrossEntropy, Huber)
//! - Training infrastructure with metrics and learning rate scheduling

use ndarray::{Array, IxDyn};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use ndarray_rand::RandomExt;
use scirs2_neural::prelude::*;

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🧠 SciRS2 Neural Network Enhanced Demo");
    println!("=====================================");

    // Initialize random number generator
    let mut rng = ndarray_rand::rand::rng();

    // Example 1: Dense Layer Classification
    println!("\n1. Dense Layer Binary Classification");
    println!("------------------------------------");
    dense_classification_demo(&mut rng)?;

    // Example 2: Conv2D Layer for Computer Vision
    println!("\n2. Conv2D Layer for Computer Vision");
    println!("-----------------------------------");
    conv2d_demo(&mut rng)?;

    // Example 3: LSTM for Sequence Modeling
    println!("\n3. LSTM for Sequence Modeling");
    println!("-----------------------------");
    lstm_demo(&mut rng)?;

    // Example 4: Complete Training Pipeline
    println!("\n4. Complete Training Pipeline");
    println!("-----------------------------");
    training_pipeline_demo(&mut rng)?;

    // Example 5: Loss Functions Comparison
    println!("\n5. Loss Functions Comparison");
    println!("----------------------------");
    loss_functions_demo()?;

    println!("\n🎉 All demos completed successfully!");
    println!("The enhanced scirs2-neural module now includes:");
    println!("  ✅ Dense, Conv2D, and LSTM layers");
    println!("  ✅ Complete activation function suite");
    println!("  ✅ Multiple loss functions");
    println!("  ✅ Training infrastructure with metrics");
    println!("  ✅ Learning rate scheduling");

    Ok(())
}

#[allow(dead_code)]
fn dense_classification_demo(rng: &mut impl ndarray_rand::rand::Rng) -> Result<()> {
    // Create a simple binary classification dataset
    let num_samples = 100;
    let num_features = 4;

    // Generate random data
    let normal = Normal::new(0.0, 1.0).unwrap();
    let input_data = Array::random_using((num_samples, num_features), normal, rng);

    // Create simple binary labels based on sum of features
    let mut labels = Array::zeros(IxDyn(&[num_samples, 1]));
    for i in 0..num_samples {
        let sum: f32 = (0..num_features).map(|j| input_data[[i, j]]).sum();
        labels[[i, 0]] = if sum > 0.0 { 1.0 } else { 0.0 };
    }

    // Create a simple neural network
    let mut model = Sequential::new();
    model.add(Dense::new(num_features, 8, Some("relu"), rng)?);
    model.add(Dropout::new(0.2, rng)?);
    model.add(Dense::new(8, 4, Some("relu"), rng)?);
    model.add(Dense::new(4, 1, Some("sigmoid"), rng)?);

    println!("Model created with {} parameters", model.total_parameters());

    // Test forward pass
    let output = model.forward(&input_data.into_dyn())?;
    println!(
        "Forward pass successful. Output shape: {:?}",
        output.shape()
    );

    // Test loss computation
    let loss_fn = BinaryCrossEntropyLoss::new();
    let loss_value = loss_fn.forward(&output, &labels)?;
    println!("Loss computation successful. Loss: {:.4}", loss_value);

    Ok(())
}

#[allow(dead_code)]
fn conv2d_demo(rng: &mut impl ndarray_rand::rand::Rng) -> Result<()> {
    // Create a simple 2D image-like dataset
    let batch_size = 4;
    let channels = 3;
    let height = 32;
    let width = 32;

    // Generate random image data
    let uniform = Uniform::new(0.0, 1.0);
    let image_data = Array::random_using((batch_size, channels, height, width), uniform, rng);

    // Create a simple CNN
    let conv_layer = Conv2D::new(
        channels,     // in_channels
        16,           // out_channels
        (3, 3),       // kernel_size
        (1, 1),       // stride
        (1, 1),       // padding
        true,         // bias
        Some("relu"), // activation
        rng,
    )?;

    println!(
        "Conv2D layer created with {} parameters",
        conv_layer.parameter_count()
    );

    // Test forward pass
    let output = conv_layer.forward(&image_data.into_dyn())?;
    println!(
        "Conv2D forward pass successful. Output shape: {:?}",
        output.shape()
    );

    // Expected output shape: [4, 16, 32, 32] (same height/width due to padding)
    let expected_shape = &[batch_size, 16, height, width];
    assert_eq!(output.shape(), expected_shape);
    println!("Output shape verification passed!");

    Ok(())
}

#[allow(dead_code)]
fn lstm_demo(rng: &mut impl ndarray_rand::rand::Rng) -> Result<()> {
    // Create a simple sequence dataset
    let batch_size = 3;
    let sequence_length = 10;
    let input_size = 5;
    let hidden_size = 8;

    // Generate random sequence data
    let normal = Normal::new(0.0, 0.5).unwrap();
    let sequence_data = Array::random_using((batch_size, sequence_length, input_size), normal, rng);

    // Create LSTM layer
    let lstm = LSTM::new(input_size, hidden_size, true, rng)?;

    println!(
        "LSTM layer created with {} parameters",
        lstm.parameter_count()
    );

    // Test forward pass with sequences
    let output = lstm.forward(&sequence_data.into_dyn())?;
    println!(
        "LSTM forward pass successful. Output shape: {:?}",
        output.shape()
    );

    // Expected output shape: [3, 10, 8]
    let expected_shape = &[batch_size, sequence_length, hidden_size];
    assert_eq!(output.shape(), expected_shape);
    println!("LSTM sequence processing verification passed!");

    // Test single time step
    let single_step = Array::random_using((batch_size, input_size), normal, rng);
    let single_output = lstm.forward(&single_step.into_dyn())?;
    println!("LSTM single step output shape: {:?}", single_output.shape());

    Ok(())
}

#[allow(dead_code)]
fn training_pipeline_demo(rng: &mut impl ndarray_rand::rand::Rng) -> Result<()> {
    // Create a simple regression dataset
    let num_samples = 200;
    let num_features = 3;

    // Generate random input data
    let uniform = Uniform::new(-2.0, 2.0);
    let input_data = Array::random_using((num_samples, num_features), uniform, rng);

    // Create targets: y = x1^2 + x2*x3 + noise
    let mut targets = Array::zeros(IxDyn(&[num_samples, 1]));
    let noise = Normal::new(0.0, 0.1).unwrap();
    for i in 0..num_samples {
        let x1 = input_data[[i, 0]];
        let x2 = input_data[[i, 1]];
        let x3 = input_data[[i, 2]];
        let y = x1 * x1 + x2 * x3 + noise.sample(rng);
        targets[[i, 0]] = y;
    }

    // Split into train and validation sets
    let train_size = (num_samples as f32 * 0.8) as usize;
    let train_x = input_data
        .slice(ndarray::s![..train_size, ..])
        .to_owned()
        .into_dyn();
    let train_y = targets
        .slice(ndarray::s![..train_size, ..])
        .to_owned()
        .into_dyn();
    let val_x = input_data
        .slice(ndarray::s![train_size.., ..])
        .to_owned()
        .into_dyn();
    let val_y = targets
        .slice(ndarray::s![train_size.., ..])
        .to_owned()
        .into_dyn();

    // Create model
    let mut model = Sequential::new();
    model.add(Dense::new(num_features, 16, Some("relu"), rng)?);
    model.add(Dropout::new(0.1, rng)?);
    model.add(Dense::new(16, 8, Some("relu"), rng)?);
    model.add(Dense::new(8, 1, None, rng)?); // No activation for regression

    // Create loss function
    let loss_fn = MSELoss::new();

    // Configure training
    let config = TrainingConfig {
        epochs: 20,
        learning_rate: 0.01,
        batch_size: 16,
        shuffle: true,
        validation_frequency: 5,
        early_stopping_patience: Some(10),
        verbose: true,
    };

    // Create trainer and train
    let mut trainer = Trainer::new(config);
    trainer.train(
        &mut model,
        &loss_fn,
        &train_x,
        &train_y,
        Some(&val_x),
        Some(&val_y),
    )?;

    // Evaluate on validation set
    let (final_loss, final_accuracy) = trainer.evaluate(&mut model, &loss_fn, &val_x, &val_y)?;
    println!("Final validation loss: {:.4}", final_loss);
    println!("Final validation accuracy: {:.4}", final_accuracy);

    // Display training metrics
    let metrics = trainer.metrics();
    println!(
        "Training completed with {} epochs",
        metrics.train_losses.len()
    );
    if let Some(best_loss) = metrics.best_val_loss() {
        println!("Best validation loss: {:.4}", best_loss);
    }

    Ok(())
}

#[allow(dead_code)]
fn loss_functions_demo() -> Result<()> {
    // Create sample predictions and targets
    let predictions = Array::from_shape_vec(
        IxDyn(&[4, 3]),
        vec![0.8, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2, 0.7, 0.3, 0.3, 0.4],
    )
    .unwrap();

    let targets = Array::from_shape_vec(
        IxDyn(&[4, 3]),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    )
    .unwrap();

    // Test different loss functions
    let loss_functions: Vec<Box<dyn Loss<f32>>> = vec![
        Box::new(MSELoss::new()),
        Box::new(CrossEntropyLoss::new()),
        Box::new(HuberLoss::new()),
    ];

    for loss_fn in loss_functions {
        let loss_value = loss_fn.forward(&predictions, &targets)?;
        let gradient = loss_fn.backward(&predictions, &targets)?;

        println!(
            "{}: loss={:.4}, grad_norm={:.4}",
            loss_fn.name(),
            loss_value,
            gradient.iter().map(|&x| x * x).sum::<f32>().sqrt()
        );
    }

    // Test binary classification loss
    let binary_predictions =
        Array::from_shape_vec(IxDyn(&[4, 1]), vec![0.8, 0.3, 0.9, 0.1]).unwrap();

    let binary_targets = Array::from_shape_vec(IxDyn(&[4, 1]), vec![1.0, 0.0, 1.0, 0.0]).unwrap();

    let binary_loss = BinaryCrossEntropyLoss::new();
    let binary_loss_value = binary_loss.forward(&binary_predictions, &binary_targets)?;
    println!("BinaryCrossEntropyLoss: loss={:.4}", binary_loss_value);

    Ok(())
}

#[allow(dead_code)]
fn learning_rate_scheduler_demo() {
    println!("\nLearning Rate Scheduler Demo:");

    let step_scheduler = StepLR::new(10, 0.5);
    let exp_scheduler = ExponentialLR::new(0.95);

    let initial_lr = 0.01;

    for epoch in 0..25 {
        let step_lr = step_scheduler.get_lr(epoch, initial_lr);
        let exp_lr = exp_scheduler.get_lr(epoch, initial_lr);

        if epoch % 5 == 0 {
            println!(
                "Epoch {}: StepLR={:.6}, ExpLR={:.6}",
                epoch, step_lr, exp_lr
            );
        }
    }
}
