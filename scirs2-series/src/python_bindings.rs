//! Python bindings for scirs2-series using PyO3
//!
//! This module provides Python bindings for seamless integration with pandas,
//! statsmodels, and other Python time series analysis libraries.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::types::{PyAny, PyDict, PyFloat, PyList, PyType};

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};

#[cfg(feature = "python")]
use ndarray::{Array1, Array2};

#[cfg(feature = "python")]
use std::collections::HashMap;

#[cfg(feature = "python")]
use crate::{
    arima_models::{ArimaConfig, ArimaModel},
    error::{Result, TimeSeriesError},
    utils::*,
};

/// Python wrapper for time series data
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTimeSeries {
    values: Array1<f64>,
    timestamps: Option<Array1<f64>>,
    frequency: Option<f64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTimeSeries {
    /// Create a new time series from Python list or numpy array
    #[new]
    fn new(
        values: PyReadonlyArray1<f64>,
        timestamps: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let values_array = values.as_array().to_owned();
        let timestamps_array = timestamps.map(|ts| ts.as_array().to_owned());

        Ok(PyTimeSeries {
            values: values_array,
            timestamps: timestamps_array,
            frequency: None,
        })
    }

    /// Set the frequency of the time series
    fn set_frequency(&mut self, frequency: f64) {
        self.frequency = Some(frequency);
    }

    /// Get the length of the time series
    fn __len__(&self) -> usize {
        self.values.len()
    }

    /// Get values as numpy array
    fn get_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values.to_pyarray_bound(py)
    }

    /// Get timestamps as numpy array (if available)
    fn get_timestamps<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.timestamps.as_ref().map(|ts| ts.to_pyarray_bound(py))
    }

    /// Convert to pandas-compatible dictionary
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("values", self.values.to_pyarray_bound(py))?;

        if let Some(ref timestamps) = self.timestamps {
            dict.set_item("timestamps", timestamps.to_pyarray_bound(py))?;
        }

        if let Some(freq) = self.frequency {
            dict.set_item("frequency", freq)?;
        }

        Ok(dict.into())
    }

    /// Create from pandas Series
    #[classmethod]
    fn from_pandas(_cls: &Bound<'_, PyType>, series: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Extract values from pandas Series
        let values = series.getattr("values")?;
        let values_array: PyReadonlyArray1<f64> = values.extract()?;

        // Try to extract index (timestamps) if available
        let index = series.getattr("index")?;
        let timestamps = if index.hasattr("values")? {
            let index_values: PyResult<PyReadonlyArray1<f64>> = index.getattr("values")?.extract();
            index_values.ok()
        } else {
            None
        };

        Self::new(values_array, timestamps)
    }

    /// Statistical summary
    fn describe(&self) -> PyResult<HashMap<String, f64>> {
        let mut stats = HashMap::new();
        let values = &self.values;

        let mean = values.mean().unwrap_or(0.0);
        let std = values.std(0.0);
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        stats.insert("count".to_string(), values.len() as f64);
        stats.insert("mean".to_string(), mean);
        stats.insert("std".to_string(), std);
        stats.insert("min".to_string(), min);
        stats.insert("max".to_string(), max);

        // Calculate quantiles
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted_values.len();

        stats.insert("25%".to_string(), sorted_values[n / 4]);
        stats.insert("50%".to_string(), sorted_values[n / 2]);
        stats.insert("75%".to_string(), sorted_values[3 * n / 4]);

        Ok(stats)
    }
}

/// Python wrapper for ARIMA models
#[cfg(feature = "python")]
#[pyclass]
pub struct PyARIMA {
    model: Option<ArimaModel<f64>>,
    config: ArimaConfig,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyARIMA {
    /// Create a new ARIMA model
    #[new]
    fn new(p: usize, d: usize, q: usize) -> Self {
        let config = ArimaConfig {
            p,
            d,
            q,
            seasonal_p: 0,
            seasonal_d: 0,
            seasonal_q: 0,
            seasonal_period: 0,
        };

        PyARIMA {
            model: None,
            config,
        }
    }

    /// Create SARIMA model
    #[classmethod]
    fn sarima(
        _cls: &Bound<'_, PyType>,
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        seasonal_period: usize,
    ) -> Self {
        let config = ArimaConfig {
            p,
            d,
            q,
            seasonal_p,
            seasonal_d,
            seasonal_q,
            seasonal_period,
        };

        PyARIMA {
            model: None,
            config,
        }
    }

    /// Fit the ARIMA model
    fn fit(&mut self, data: &PyTimeSeries) -> PyResult<()> {
        let mut model = ArimaModel::new(self.config.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        model
            .fit(&data.values)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        self.model = Some(model);
        Ok(())
    }

    /// Generate forecasts
    fn forecast<'py>(&self, py: Python<'py>, steps: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match &self.model {
            Some(model) => {
                let forecasts = model.forecast(steps).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                })?;
                Ok(forecasts.to_pyarray_bound(py))
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model not fitted. Call fit() first.",
            )),
        }
    }

    /// Get model parameters
    fn get_params(&self) -> PyResult<HashMap<String, f64>> {
        match &self.model {
            Some(model) => {
                let params = model.get_params().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                })?;
                Ok(params)
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model not fitted. Call fit() first.",
            )),
        }
    }

    /// Get model summary (similar to statsmodels)
    fn summary(&self) -> PyResult<String> {
        match &self.model {
            Some(model) => {
                let params = model.get_params().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                })?;

                let mut summary = format!(
                    "ARIMA({},{},{}) Model Results\n",
                    self.config.p, self.config.d, self.config.q
                );
                summary.push_str("=====================================\n");

                for (param, value) in params {
                    summary.push_str(&format!("{:20}: {:10.6}\n", param, value));
                }

                Ok(summary)
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model not fitted. Call fit() first.",
            )),
        }
    }
}

/// Utility functions for Python integration
#[cfg(feature = "python")]
#[pyfunction]
fn calculate_statistics(data: &PyTimeSeries) -> PyResult<HashMap<String, f64>> {
    let stats = calculate_basic_stats(&data.values)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    Ok(stats)
}

#[cfg(feature = "python")]
#[pyfunction]
fn check_stationarity(data: &PyTimeSeries) -> PyResult<bool> {
    let stationary = is_stationary(&data.values)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    Ok(stationary)
}

#[cfg(feature = "python")]
#[pyfunction]
fn apply_differencing<'py>(
    py: Python<'py>,
    data: &PyTimeSeries,
    periods: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let differenced = difference_series(&data.values, periods)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    Ok(differenced.to_pyarray_bound(py))
}

#[cfg(feature = "python")]
#[pyfunction]
fn apply_seasonal_differencing<'py>(
    py: Python<'py>,
    data: &PyTimeSeries,
    periods: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let differenced = seasonal_difference_series(&data.values, periods)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    Ok(differenced.to_pyarray_bound(py))
}

/// Auto-ARIMA functionality for Python
#[cfg(feature = "python")]
#[pyfunction]
fn auto_arima(
    data: &PyTimeSeries,
    max_p: usize,
    max_d: usize,
    max_q: usize,
    seasonal: bool,
    max_seasonal_p: Option<usize>,
    max_seasonal_d: Option<usize>,
    max_seasonal_q: Option<usize>,
    seasonal_period: Option<usize>,
) -> PyResult<PyARIMA> {
    let max_sp = max_seasonal_p.unwrap_or(0);
    let max_sd = max_seasonal_d.unwrap_or(0);
    let max_sq = max_seasonal_q.unwrap_or(0);
    let s_period = seasonal_period.unwrap_or(0);

    let best_config = crate::arima_models::auto_arima(
        &data.values,
        max_p,
        max_d,
        max_q,
        seasonal,
        max_sp,
        max_sd,
        max_sq,
        s_period,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

    Ok(PyARIMA {
        model: None,
        config: best_config,
    })
}

/// Python module definition
#[cfg(feature = "python")]
#[pymodule]
fn scirs2_series(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimeSeries>()?;
    m.add_class::<PyARIMA>()?;

    m.add_function(wrap_pyfunction!(calculate_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(check_stationarity, m)?)?;
    m.add_function(wrap_pyfunction!(apply_differencing, m)?)?;
    m.add_function(wrap_pyfunction!(apply_seasonal_differencing, m)?)?;
    m.add_function(wrap_pyfunction!(auto_arima, m)?)?;

    Ok(())
}

// Helper functions for pandas integration
#[cfg(feature = "python")]
pub fn create_pandas_dataframe(
    py: Python,
    data: HashMap<String, Array1<f64>>,
) -> PyResult<PyObject> {
    let pandas = py.import_bound("pandas")?;
    let dict = PyDict::new_bound(py);

    for (key, values) in data {
        dict.set_item(key, values.to_pyarray_bound(py))?;
    }

    let df = pandas.call_method1("DataFrame", (dict,))?;
    Ok(df.into())
}

#[cfg(feature = "python")]
pub fn create_pandas_series(
    py: Python,
    data: Array1<f64>,
    name: Option<&str>,
) -> PyResult<PyObject> {
    let pandas = py.import_bound("pandas")?;
    let args = (data.to_pyarray_bound(py),);
    let kwargs = PyDict::new_bound(py);

    if let Some(name) = name {
        kwargs.set_item("name", name)?;
    }

    let series = pandas.call_method("Series", args, Some(&kwargs))?;
    Ok(series.into())
}
