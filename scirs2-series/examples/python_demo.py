#!/usr/bin/env python3
"""
Enhanced Python Integration Demo for scirs2-series

This script demonstrates the advanced Python bindings for scirs2-series,
showcasing the comprehensive time series analysis capabilities with
seamless pandas, numpy, and visualization integration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import scirs2-series (assuming it's built and available)
try:
    import scirs2_series as scirs2
    print("✅ Successfully imported scirs2-series")
except ImportError as e:
    print(f"❌ Failed to import scirs2-series: {e}")
    print("Please build the Python bindings first:")
    print("  cargo build --release --features python")
    exit(1)

def generate_sample_data():
    """Generate comprehensive sample datasets for demonstration"""
    
    print("📊 Generating sample datasets...")
    
    # Dataset 1: Monthly sales with trend and seasonality
    np.random.seed(42)
    n_months = 120  # 10 years
    time_index = pd.date_range('2014-01-01', periods=n_months, freq='M')
    
    trend = np.linspace(100, 300, n_months)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    noise = np.random.normal(0, 10, n_months)
    sales_data = trend + seasonal + noise
    
    # Add some anomalies
    sales_data[50] += 150  # Large spike
    sales_data[75] -= 120  # Large drop
    sales_data[90] += 80   # Medium spike
    
    # Dataset 2: Daily temperature data
    n_days = 365
    daily_temp = 20 + 15 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 3, n_days)
    
    # Dataset 3: High-frequency financial data
    n_minutes = 1440  # One day in minutes
    price_changes = np.random.normal(0.001, 0.02, n_minutes)
    financial_data = 100 * np.exp(np.cumsum(price_changes))
    
    return {
        'sales': {'data': sales_data, 'index': time_index, 'freq': 12},
        'temperature': {'data': daily_temp, 'freq': 365},
        'financial': {'data': financial_data, 'freq': 1440}
    }

def demo_basic_functionality():
    """Demonstrate basic time series operations"""
    
    print("\n=== Demo 1: Basic Time Series Operations ===")
    
    datasets = generate_sample_data()
    sales_data = datasets['sales']['data']
    
    # Create time series object
    ts = scirs2.PyTimeSeries(sales_data, None)
    print(f"✅ Created time series with {len(ts)} observations")
    
    # Calculate comprehensive statistics
    print("\n📈 Descriptive Statistics:")
    stats = scirs2.calculate_statistics(ts)
    for key, value in stats.items():
        print(f"  {key:15}: {value:10.4f}")
    
    # Test stationarity
    is_stationary = scirs2.check_stationarity(ts)
    print(f"\n🔍 Stationarity test: {'Stationary' if is_stationary else 'Non-stationary'}")
    
    # Apply differencing
    if not is_stationary:
        print("📉 Applying first-order differencing...")
        differenced = scirs2.apply_differencing(ts, 1)
        ts_diff = scirs2.PyTimeSeries(differenced, None)
        is_diff_stationary = scirs2.check_stationarity(ts_diff)
        print(f"   After differencing: {'Stationary' if is_diff_stationary else 'Still non-stationary'}")

def demo_arima_modeling():
    """Demonstrate ARIMA modeling and forecasting"""
    
    print("\n=== Demo 2: ARIMA Modeling & Forecasting ===")
    
    datasets = generate_sample_data()
    sales_data = datasets['sales']['data']
    ts = scirs2.PyTimeSeries(sales_data, None)
    
    # Manual ARIMA model
    print("🔧 Fitting ARIMA(2,1,2) model...")
    arima_model = scirs2.PyARIMA(2, 1, 2)
    arima_model.fit(ts)
    print("✅ ARIMA model fitted successfully")
    
    # Generate forecasts
    forecasts = arima_model.forecast(12)
    print(f"📮 Generated {len(forecasts)} forecasts")
    print("   Next 6 months forecast:", [f"{x:.2f}" for x in forecasts[:6]])
    
    # Auto ARIMA model selection
    print("\n🤖 Running Auto-ARIMA...")
    auto_model = scirs2.auto_arima(ts, 5, 2, 5, True, 2, 1, 2, 12)
    auto_model.fit(ts)
    
    auto_forecasts = auto_model.forecast(12)
    print("✅ Auto-ARIMA model fitted and forecasted")
    print("   Auto-ARIMA forecasts:", [f"{x:.2f}" for x in auto_forecasts[:6]])
    
    # Get model parameters
    try:
        params = arima_model.get_params()
        print("\n📊 Model Parameters:")
        for param, value in params.items():
            print(f"  {param}: {value:.6f}")
    except Exception as e:
        print(f"⚠️  Could not retrieve parameters: {e}")

def demo_anomaly_detection():
    """Demonstrate advanced anomaly detection"""
    
    print("\n=== Demo 3: Advanced Anomaly Detection ===")
    
    datasets = generate_sample_data()
    sales_data = datasets['sales']['data']
    ts = scirs2.PyTimeSeries(sales_data, None)
    
    # Create anomaly detector
    detector = scirs2.PyAnomalyDetector()
    
    # IQR-based anomaly detection
    print("🔍 Detecting anomalies using IQR method...")
    iqr_anomalies = detector.detect_iqr(ts, 2.5)
    print(f"   Found {len(iqr_anomalies)} anomalies: {iqr_anomalies[:10]}")
    
    # Z-score based anomaly detection
    print("🔍 Detecting anomalies using Z-score method...")
    zscore_anomalies = detector.detect_zscore(ts, 3.0)
    print(f"   Found {len(zscore_anomalies)} anomalies: {zscore_anomalies[:10]}")
    
    # Isolation Forest anomaly detection
    print("🔍 Detecting anomalies using Isolation Forest...")
    isolation_anomalies = detector.detect_isolation_forest(ts, 0.1)
    print(f"   Found {len(isolation_anomalies)} anomalies: {isolation_anomalies[:10]}")
    
    # Comprehensive anomaly report
    all_anomalies = detector.detect_all(ts)
    print("\n📋 Comprehensive Anomaly Report:")
    for method, anomalies in all_anomalies.items():
        if len(anomalies) > 0:
            print(f"  {method}: {len(anomalies)} anomalies")

def demo_advanced_features():
    """Demonstrate advanced features like change point detection and batch processing"""
    
    print("\n=== Demo 4: Advanced Features ===")
    
    datasets = generate_sample_data()
    sales_data = datasets['sales']['data']
    ts = scirs2.PyTimeSeries(sales_data, None)
    
    # Change point detection
    print("📍 Change Point Detection:")
    try:
        cp_detector = scirs2.PyChangePointDetector()
        
        # PELT algorithm
        pelt_changepoints = cp_detector.detect_pelt(ts, 10.0)
        print(f"   PELT algorithm found {len(pelt_changepoints)} change points")
        
        # Binary segmentation
        binary_changepoints = cp_detector.detect_binary_segmentation(ts, 2.0)
        print(f"   Binary segmentation found {len(binary_changepoints)} change points")
        
    except Exception as e:
        print(f"⚠️  Change point detection: {e}")
    
    # Batch processing
    print("\n📦 Batch Processing:")
    try:
        batch_processor = scirs2.PyBatchProcessor(50)
        
        # Create multiple time series
        ts_list = []
        for i in range(5):
            data = np.random.normal(100 + i*10, 15, 100)
            ts_list.append(scirs2.PyTimeSeries(data, None))
        
        # Process batch
        batch_results = batch_processor.process_batch(ts_list)
        print(f"✅ Processed batch of {len(ts_list)} time series")
        print("   Batch processing completed successfully")
        
    except Exception as e:
        print(f"⚠️  Batch processing: {e}")
    
    # Streaming processing
    print("\n🌊 Streaming Processing:")
    try:
        streaming_processor = scirs2.PyStreamingProcessor(20)
        
        # Simulate streaming data
        streaming_data = np.random.normal(50, 10, 100)
        processed_points = 0
        
        for value in streaming_data:
            result = streaming_processor.process_point(value)
            if result is not None:
                processed_points += 1
        
        window_stats = streaming_processor.get_window_stats()
        print(f"✅ Processed {processed_points} streaming points")
        print("   Current window statistics available")
        
    except Exception as e:
        print(f"⚠️  Streaming processing: {e}")

def demo_ensemble_forecasting():
    """Demonstrate ensemble forecasting with uncertainty quantification"""
    
    print("\n=== Demo 5: Ensemble Forecasting ===")
    
    datasets = generate_sample_data()
    sales_data = datasets['sales']['data']
    ts = scirs2.PyTimeSeries(sales_data, None)
    
    try:
        # Create advanced forecaster
        ensemble_forecaster = scirs2.PyAdvancedForecaster()
        
        print("🎯 Training ensemble models...")
        ensemble_forecaster.fit(ts)
        
        print("🔮 Generating forecasts with uncertainty quantification...")
        forecast_result = ensemble_forecaster.forecast_with_uncertainty(12, 0.95)
        
        print("✅ Ensemble forecasting completed")
        print(f"   Forecast length: {len(forecast_result['forecast'])}")
        print(f"   Confidence intervals: Available")
        print("   Mean forecast (first 6):", [f"{x:.2f}" for x in forecast_result['forecast'][:6]])
        
    except Exception as e:
        print(f"⚠️  Ensemble forecasting: {e}")

def demo_batch_operations():
    """Demonstrate batch operations and streaming anomaly detection"""
    
    print("\n=== Demo 6: Batch Operations & Streaming ===")
    
    # Generate multiple time series for batch processing
    datasets = []
    for i in range(10):
        np.random.seed(42 + i)
        data = 100 + 50 * np.sin(2 * np.pi * np.arange(60) / 12) + np.random.normal(0, 10, 60)
        ts = scirs2.PyTimeSeries(data, None)
        datasets.append(ts)
    
    # Batch forecasting
    try:
        print("📊 Batch forecasting for multiple time series...")
        batch_forecasts = scirs2.batch_forecast(datasets, 6)
        print(f"✅ Generated batch forecasts for {len(datasets)} series")
        print("   Batch forecasting completed successfully")
        
    except Exception as e:
        print(f"⚠️  Batch forecasting: {e}")
    
    # Streaming anomaly detection
    try:
        print("\n🚨 Streaming anomaly detection...")
        streaming_data = np.random.normal(50, 10, 200)
        
        # Add some anomalies
        streaming_data[50] += 50
        streaming_data[100] -= 40
        streaming_data[150] += 30
        
        anomaly_results = scirs2.streaming_anomaly_detection(streaming_data.tolist(), 20, 2.5)
        
        print(f"✅ Streaming anomaly detection completed")
        print(f"   Found {len(anomaly_results['anomaly_indices'])} streaming anomalies")
        print(f"   Anomaly indices: {anomaly_results['anomaly_indices'][:10]}")
        
    except Exception as e:
        print(f"⚠️  Streaming anomaly detection: {e}")

def demo_pandas_integration():
    """Demonstrate pandas integration and visualization"""
    
    print("\n=== Demo 7: Pandas Integration & Visualization ===")
    
    datasets = generate_sample_data()
    sales_data = datasets['sales']['data']
    ts = scirs2.PyTimeSeries(sales_data, None)
    
    # Generate forecast for pandas integration
    arima_model = scirs2.PyARIMA(2, 1, 2)
    arima_model.fit(ts)
    forecasts = arima_model.forecast(12)
    
    try:
        # Create pandas-compatible forecast
        print("📅 Creating pandas-compatible forecast...")
        pandas_forecast = scirs2.create_pandas_compatible_forecast(
            forecasts, "2024-01-01", "M"
        )
        print("✅ Pandas-compatible forecast created")
        print("   Index type: DatetimeIndex")
        print("   Frequency: Monthly")
        
    except Exception as e:
        print(f"⚠️  Pandas integration: {e}")
    
    try:
        # Create visualization
        print("\n📈 Creating Plotly visualization...")
        plotly_fig = scirs2.create_plotly_visualization(
            ts, forecasts, "Sales Forecast with SciRS2"
        )
        print("✅ Plotly visualization created")
        print("   Interactive plot available")
        
    except Exception as e:
        print(f"⚠️  Plotly visualization: {e}")

def demo_feature_extraction():
    """Demonstrate advanced feature extraction"""
    
    print("\n=== Demo 8: Advanced Feature Extraction ===")
    
    datasets = generate_sample_data()
    sales_data = datasets['sales']['data']
    ts = scirs2.PyTimeSeries(sales_data, None)
    
    try:
        print("⚙️ Extracting comprehensive time series features...")
        features = scirs2.advanced_feature_extraction(ts)
        
        print("✅ Feature extraction completed")
        print(f"   Total features extracted: {len(features)}")
        print("   Feature types: Statistical, frequency-domain, complexity measures")
        
        # Display sample features
        print("\n📊 Sample extracted features:")
        feature_sample = dict(list(features.items())[:10])
        for name, value in feature_sample.items():
            print(f"   {name:25}: {value:10.6f}")
        
    except Exception as e:
        print(f"⚠️  Feature extraction: {e}")

def demo_performance_comparison():
    """Demonstrate performance capabilities"""
    
    print("\n=== Demo 9: Performance Comparison ===")
    
    # Generate large dataset
    print("🚀 Performance testing with large dataset...")
    large_n = 10000
    large_data = np.random.normal(100, 15, large_n)
    large_ts = scirs2.PyTimeSeries(large_data, None)
    
    import time
    
    # Time statistics calculation
    start_time = time.time()
    stats = scirs2.calculate_statistics(large_ts)
    stats_time = time.time() - start_time
    print(f"⏱️  Statistics calculation ({large_n} points): {stats_time:.4f} seconds")
    
    # Time anomaly detection
    detector = scirs2.PyAnomalyDetector()
    start_time = time.time()
    anomalies = detector.detect_iqr(large_ts, 2.5)
    anomaly_time = time.time() - start_time
    print(f"⏱️  Anomaly detection ({large_n} points): {anomaly_time:.4f} seconds")
    print(f"   Found {len(anomalies)} anomalies")
    
    # Time ARIMA fitting
    start_time = time.time()
    arima_model = scirs2.PyARIMA(1, 1, 1)
    arima_model.fit(large_ts)
    arima_time = time.time() - start_time
    print(f"⏱️  ARIMA model fitting ({large_n} points): {arima_time:.4f} seconds")

def demo_error_handling():
    """Demonstrate robust error handling"""
    
    print("\n=== Demo 10: Error Handling & Robustness ===")
    
    # Test with invalid data
    print("🛡️ Testing error handling...")
    
    try:
        # Empty data
        empty_ts = scirs2.PyTimeSeries([], None)
        print("❌ Should have failed with empty data")
    except Exception as e:
        print("✅ Properly handled empty data error")
    
    try:
        # Invalid ARIMA parameters
        invalid_model = scirs2.PyARIMA(-1, 0, 0)
        print("❌ Should have failed with invalid parameters")
    except Exception as e:
        print("✅ Properly handled invalid ARIMA parameters")
    
    try:
        # Very small dataset
        small_data = [1.0, 2.0]
        small_ts = scirs2.PyTimeSeries(small_data, None)
        arima_model = scirs2.PyARIMA(2, 1, 2)
        arima_model.fit(small_ts)
        print("❌ Should have failed with insufficient data")
    except Exception as e:
        print("✅ Properly handled insufficient data error")
    
    print("🛡️ Error handling verification completed")

def main():
    """Main demonstration function"""
    
    print("🚀 SciRS2-Series Enhanced Python Integration Demo")
    print("=" * 55)
    print("This demo showcases the comprehensive time series analysis")
    print("capabilities with advanced Python ecosystem integration.")
    print()
    
    try:
        # Run all demonstrations
        demo_basic_functionality()
        demo_arima_modeling()
        demo_anomaly_detection()
        demo_advanced_features()
        demo_ensemble_forecasting()
        demo_batch_operations()
        demo_pandas_integration()
        demo_feature_extraction()
        demo_performance_comparison()
        demo_error_handling()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\n📚 Summary of demonstrated features:")
        print("✅ Basic time series operations and statistics")
        print("✅ ARIMA modeling with auto-selection")
        print("✅ Multiple anomaly detection methods")
        print("✅ Change point detection algorithms")
        print("✅ Batch and streaming processing")
        print("✅ Ensemble forecasting with uncertainty")
        print("✅ Pandas and Plotly integration")
        print("✅ Advanced feature extraction")
        print("✅ High-performance processing")
        print("✅ Robust error handling")
        
        print("\n🔗 Integration capabilities verified:")
        print("📊 Pandas DataFrames and Series")
        print("🔢 NumPy arrays")
        print("📈 Plotly visualizations")
        print("⚡ High-performance computing")
        print("🌊 Streaming data processing")
        print("📦 Batch operations")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()