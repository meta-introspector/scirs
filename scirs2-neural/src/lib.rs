//! Enhanced scirs2-neural implementation with core activation functions

pub mod error;
pub mod activations_minimal;
pub mod layers;
pub mod losses;
pub mod training;

pub use error::{Error, NeuralError, Result};
pub use activations_minimal::{Activation, GELU, Tanh, Sigmoid, ReLU, Softmax};
pub use layers::{Layer, Dense, Dropout, Conv2D, LSTM, Sequential};
pub use losses::{Loss, MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, HuberLoss, Reduction};
pub use training::{Trainer, TrainingConfig, TrainingMetrics, LearningRateScheduler, StepLR, ExponentialLR};

/// Working prelude with core functionality
pub mod prelude {
    pub use crate::{
        activations_minimal::{Activation, GELU, Tanh, Sigmoid, ReLU, Softmax},
        error::{Error, NeuralError, Result},
        layers::{Layer, Dense, Dropout, Conv2D, LSTM, Sequential},
        losses::{Loss, MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, HuberLoss, Reduction},
        training::{Trainer, TrainingConfig, TrainingMetrics, LearningRateScheduler, StepLR, ExponentialLR},
    };
}