//! WebAssembly bindings for scirs2-series
//!
//! This module provides JavaScript bindings for time series analysis functionality,
//! enabling browser-based time series analysis with full performance and feature parity.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use js_sys::Array;

#[cfg(feature = "wasm")]
use web_sys::console;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "wasm")]
use crate::{
    anomaly::{detect_anomalies, AnomalyMethod, AnomalyOptions},
    arima_models::{ArimaModel, ArimaSelectionOptions, SarimaParams},
    decomposition::{stl_decomposition, STLOptions},
    error::Result,
    forecasting::neural::NeuralForecaster,
    utils::*,
};

#[cfg(feature = "wasm")]
use ndarray::{Array1, Array2};

#[cfg(feature = "wasm")]
use std::collections::HashMap;

// Utility macro for error handling in WASM
#[cfg(feature = "wasm")]
macro_rules! js_error {
    ($msg:expr) => {
        JsValue::from_str(&format!("Error: {}", $msg))
    };
}

#[cfg(feature = "wasm")]
macro_rules! js_result {
    ($result:expr) => {
        match $result {
            Ok(val) => Ok(val),
            Err(e) => Err(js_error!(e)),
        }
    };
}

/// Time series data structure for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    values: Vec<f64>,
    timestamps: Option<Vec<f64>>,
    frequency: Option<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl TimeSeriesData {
    /// Create a new time series from JavaScript array
    #[wasm_bindgen(constructor)]
    pub fn new(values: &[f64]) -> TimeSeriesData {
        TimeSeriesData {
            values: values.to_vec(),
            timestamps: None,
            frequency: None,
        }
    }

    /// Create time series with timestamps
    #[wasm_bindgen]
    pub fn with_timestamps(values: &[f64], timestamps: &[f64]) -> Result<TimeSeriesData, JsValue> {
        if values.len() != timestamps.len() {
            return Err(js_error!("Values and timestamps must have the same length"));
        }

        Ok(TimeSeriesData {
            values: values.to_vec(),
            timestamps: Some(timestamps.to_vec()),
            frequency: None,
        })
    }

    /// Set frequency for the time series
    #[wasm_bindgen]
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = Some(frequency);
    }

    /// Get the length of the time series
    #[wasm_bindgen]
    pub fn length(&self) -> usize {
        self.values.len()
    }

    /// Get values as JavaScript array
    #[wasm_bindgen]
    pub fn get_values(&self) -> Vec<f64> {
        self.values.clone()
    }

    /// Get timestamps as JavaScript array
    #[wasm_bindgen]
    pub fn get_timestamps(&self) -> Option<Vec<f64>> {
        self.timestamps.clone()
    }
}

/// ARIMA model wrapper for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmARIMA {
    model: Option<ArimaModel<f64>>,
    config: SarimaParams,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmARIMA {
    /// Create a new ARIMA model
    #[wasm_bindgen(constructor)]
    pub fn new(p: usize, d: usize, q: usize) -> WasmARIMA {
        let config = SarimaParams {
            pdq: (p, d, q),
            seasonal_pdq: (0, 0, 0),
            seasonal_period: 1,
        };

        WasmARIMA {
            model: None,
            config,
        }
    }

    /// Create SARIMA model with seasonal components
    #[wasm_bindgen]
    pub fn sarima(
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        seasonal_period: usize,
    ) -> WasmARIMA {
        let config = crate::forecasting::ArimaParams {
            p,
            d,
            q,
            seasonal_p,
            seasonal_d,
            seasonal_q,
            seasonal_period,
            trend: Some("c".to_string()),
            enforce_stationarity: true,
            enforce_invertibility: true,
            concentrate_scale: false,
            dates: None,
            freq: None,
            missing: "none".to_string(),
            validate_specification: true,
        };

        WasmARIMA {
            model: None,
            config,
        }
    }

    /// Fit the ARIMA model to time series data
    #[wasm_bindgen]
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<(), JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let mut model = js_result!(ARIMAModel::new(self.config.clone()))?;
        js_result!(model.fit(&arr))?;
        self.model = Some(model);
        Ok(())
    }

    /// Generate forecasts
    #[wasm_bindgen]
    pub fn forecast(&self, steps: usize) -> Result<Vec<f64>, JsValue> {
        match &self.model {
            Some(model) => {
                let forecasts = js_result!(model.forecast(steps))?;
                Ok(forecasts.to_vec())
            }
            None => Err(js_error!("Model not fitted. Call fit() first.")),
        }
    }

    /// Get model parameters
    #[wasm_bindgen]
    pub fn get_params(&self) -> Result<JsValue, JsValue> {
        match &self.model {
            Some(model) => {
                let params = js_result!(model.get_params())?;
                Ok(serde_wasm_bindgen::to_value(&params)?)
            }
            None => Err(js_error!("Model not fitted. Call fit() first.")),
        }
    }
}

/// Anomaly detector wrapper for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmAnomalyDetector {
    method: AnomalyMethod,
    options: AnomalyOptions,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmAnomalyDetector {
    /// Create a new anomaly detector
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmAnomalyDetector {
        WasmAnomalyDetector {
            method: AnomalyMethod::ZScore,
            options: AnomalyOptions::default(),
        }
    }

    /// Detect anomalies using IQR method
    #[wasm_bindgen]
    pub fn detect_iqr(
        &self,
        data: &TimeSeriesData,
        multiplier: f64,
    ) -> Result<Vec<usize>, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let anomalies = js_result!(self.detector.detect_iqr(&arr, multiplier))?;
        Ok(anomalies)
    }

    /// Detect anomalies using Z-score method
    #[wasm_bindgen]
    pub fn detect_zscore(
        &self,
        data: &TimeSeriesData,
        threshold: f64,
    ) -> Result<Vec<usize>, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let anomalies = js_result!(self.detector.detect_zscore(&arr, threshold))?;
        Ok(anomalies)
    }

    /// Detect anomalies using isolation forest
    #[wasm_bindgen]
    pub fn detect_isolation_forest(
        &self,
        data: &TimeSeriesData,
        contamination: f64,
    ) -> Result<Vec<usize>, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let anomalies = js_result!(self.detector.detect_isolation_forest(&arr, contamination))?;
        Ok(anomalies)
    }
}

/// STL decomposition wrapper for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmSTLDecomposition {
    options: STLOptions,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DecompositionResult {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmSTLDecomposition {
    /// Create a new STL decomposition
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> WasmSTLDecomposition {
        WasmSTLDecomposition {
            decomposition: STLDecomposition::new(period),
        }
    }

    /// Perform STL decomposition
    #[wasm_bindgen]
    pub fn decompose(&self, data: &TimeSeriesData) -> Result<JsValue, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let result = js_result!(self.decomposition.decompose(&arr))?;

        let decomp_result = DecompositionResult {
            trend: result.trend.to_vec(),
            seasonal: result.seasonal.to_vec(),
            residual: result.residual.to_vec(),
        };

        Ok(serde_wasm_bindgen::to_value(&decomp_result)?)
    }
}

/// Neural forecaster wrapper for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmNeuralForecaster {
    forecaster: Option<NeuralForecaster>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmNeuralForecaster {
    /// Create a new neural forecaster
    #[wasm_bindgen(constructor)]
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> WasmNeuralForecaster {
        WasmNeuralForecaster {
            forecaster: Some(NeuralForecaster::new(input_size, hidden_size, output_size)),
        }
    }

    /// Train the neural forecaster
    #[wasm_bindgen]
    pub fn train(
        &mut self,
        data: &TimeSeriesData,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<(), JsValue> {
        if let Some(forecaster) = &mut self.forecaster {
            let arr = Array1::from_vec(data.values.clone());
            js_result!(forecaster.train(&arr, epochs, learning_rate))?;
            Ok(())
        } else {
            Err(js_error!("Neural forecaster not initialized"))
        }
    }

    /// Generate forecasts using the neural model
    #[wasm_bindgen]
    pub fn forecast(&self, input: &[f64], steps: usize) -> Result<Vec<f64>, JsValue> {
        if let Some(forecaster) = &self.forecaster {
            let input_arr = Array1::from_vec(input.to_vec());
            let forecasts = js_result!(forecaster.forecast(&input_arr, steps))?;
            Ok(forecasts.to_vec())
        } else {
            Err(js_error!("Neural forecaster not initialized"))
        }
    }
}

/// Utility functions for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmUtils;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmUtils {
    /// Calculate basic statistics for time series
    #[wasm_bindgen]
    pub fn calculate_stats(data: &TimeSeriesData) -> Result<JsValue, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let stats = js_result!(calculate_basic_stats(&arr))?;
        Ok(serde_wasm_bindgen::to_value(&stats)?)
    }

    /// Check if time series is stationary
    #[wasm_bindgen]
    pub fn is_stationary(data: &TimeSeriesData) -> Result<bool, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let stationary = js_result!(is_stationary(&arr))?;
        Ok(stationary)
    }

    /// Apply differencing to make time series stationary
    #[wasm_bindgen]
    pub fn difference(data: &TimeSeriesData, periods: usize) -> Result<Vec<f64>, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let differenced = js_result!(difference_series(&arr, periods))?;
        Ok(differenced.to_vec())
    }

    /// Apply seasonal differencing
    #[wasm_bindgen]
    pub fn seasonal_difference(data: &TimeSeriesData, periods: usize) -> Result<Vec<f64>, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let differenced = js_result!(seasonal_difference_series(&arr, periods))?;
        Ok(differenced.to_vec())
    }
}

/// Initialize WASM module
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console::log_1(&"SciRS2 Time Series Analysis WASM module initialized".into());
}

/// Log function for debugging
#[cfg(feature = "wasm")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(feature = "wasm")]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

// Auto-ARIMA functionality for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmAutoARIMA;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmAutoARIMA {
    /// Automatically select best ARIMA model
    #[wasm_bindgen]
    pub fn auto_arima(
        data: &TimeSeriesData,
        max_p: usize,
        max_d: usize,
        max_q: usize,
        seasonal: bool,
        max_seasonal_p: Option<usize>,
        max_seasonal_d: Option<usize>,
        max_seasonal_q: Option<usize>,
        seasonal_period: Option<usize>,
    ) -> Result<WasmARIMA, JsValue> {
        let arr = Array1::from_vec(data.values.clone());

        let max_sp = max_seasonal_p.unwrap_or(0);
        let max_sd = max_seasonal_d.unwrap_or(0);
        let max_sq = max_seasonal_q.unwrap_or(0);
        let s_period = seasonal_period.unwrap_or(0);

        let options = crate::arima_models::ArimaSelectionOptions {
            max_p,
            max_d,
            max_q,
            seasonal,
            seasonal_period: if seasonal { seasonal_period } else { None },
            max_seasonal_p: max_sp,
            max_seasonal_d: max_sd,
            max_seasonal_q: max_sq,
            ..Default::default()
        };
        
        let (model, params) = js_result!(crate::arima_models::auto_arima(&arr, &options))?;

        Ok(WasmARIMA {
            model: Some(model),
            config: params,
        })
    }
}

// Export main functions for easier access
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn create_time_series(values: &[f64]) -> TimeSeriesData {
    TimeSeriesData::new(values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn create_arima_model(p: usize, d: usize, q: usize) -> WasmARIMA {
    WasmARIMA::new(p, d, q)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn create_anomaly_detector() -> WasmAnomalyDetector {
    WasmAnomalyDetector::new()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn create_stl_decomposition(period: usize) -> WasmSTLDecomposition {
    WasmSTLDecomposition::new(period)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn create_neural_forecaster(
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
) -> WasmNeuralForecaster {
    WasmNeuralForecaster::new(input_size, hidden_size, output_size)
}
