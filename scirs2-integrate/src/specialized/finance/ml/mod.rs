//! Machine learning applications in quantitative finance
//! 
//! This module provides machine learning models and tools specifically designed for
//! financial applications including pricing, risk management, and portfolio optimization.
//! 
//! # Modules
//! - `volatility_forecast`: Volatility prediction models
//! - `portfolio_optim`: ML-enhanced portfolio optimization
//! - `neural_pricing`: Neural network-based derivative pricing

pub mod volatility_forecast;
pub mod portfolio_optim;
pub mod neural_pricing;

// Re-export commonly used items
pub use volatility_forecast::VolatilityForecaster;
pub use portfolio_optim::MLPortfolioOptimizer;
pub use neural_pricing::NeuralPricer;

// TODO: Implement common ML utilities for finance
// - Feature engineering for financial time series
// - Market regime detection
// - Anomaly detection for risk management
// - Reinforcement learning for trading strategies