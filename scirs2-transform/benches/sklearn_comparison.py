#!/usr/bin/env python3
"""
Benchmark scikit-learn transformations for comparison with scirs2-transform.

This script runs equivalent operations to transform_bench.rs using scikit-learn
and outputs timing results in a comparable format.
"""

import time
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler,
    QuantileTransformer, PolynomialFeatures, PowerTransformer,
    Binarizer
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

# Test configurations matching Rust benchmarks
SAMPLE_SIZES = [100, 1000, 10_000]
FEATURE_SIZES = [10, 50, 100]


def benchmark_operation(name, operation, data, n_iterations=100):
    """Benchmark a single operation."""
    # Warm-up
    for _ in range(5):
        _ = operation()
    
    # Actual benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = operation()
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000
    
    elements = data.shape[0] * data.shape[1]
    throughput = elements / (mean_time / 1000)  # Elements per second
    
    print(f"{name}: {mean_time:.3f}±{std_time:.3f} ms, "
          f"Throughput: {throughput/1e6:.2f} M elements/s")


def benchmark_normalization():
    """Benchmark normalization operations."""
    print("\n=== Normalization Benchmarks ===")
    
    for n_samples in SAMPLE_SIZES:
        for n_features in FEATURE_SIZES:
            print(f"\nData shape: {n_samples}x{n_features}")
            data = np.random.uniform(-100, 100, (n_samples, n_features))
            
            # MinMax normalization
            scaler = MinMaxScaler()
            scaler.fit(data)
            benchmark_operation(
                "MinMax", 
                lambda: scaler.transform(data),
                data
            )
            
            # Z-score normalization
            scaler = StandardScaler()
            scaler.fit(data)
            benchmark_operation(
                "ZScore",
                lambda: scaler.transform(data),
                data
            )
            
            # Robust normalization
            scaler = RobustScaler()
            scaler.fit(data)
            benchmark_operation(
                "Robust",
                lambda: scaler.transform(data),
                data
            )


def benchmark_scaling():
    """Benchmark scaling operations."""
    print("\n=== Scaling Benchmarks ===")
    
    for n_samples in SAMPLE_SIZES:
        for n_features in FEATURE_SIZES:
            print(f"\nData shape: {n_samples}x{n_features}")
            data = np.random.uniform(-100, 100, (n_samples, n_features))
            
            # MaxAbsScaler
            scaler = MaxAbsScaler()
            scaler.fit(data)
            benchmark_operation(
                "MaxAbsScaler",
                lambda: scaler.transform(data),
                data
            )
            
            # QuantileTransformer
            transformer = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
            transformer.fit(data)
            benchmark_operation(
                "QuantileTransformer",
                lambda: transformer.transform(data),
                data
            )


def benchmark_feature_engineering():
    """Benchmark feature engineering operations."""
    print("\n=== Feature Engineering Benchmarks ===")
    
    for n_samples in [100, 1000]:
        for n_features in [5, 10, 20]:
            print(f"\nData shape: {n_samples}x{n_features}")
            data = np.random.uniform(-10, 10, (n_samples, n_features))
            
            # Polynomial features (degree 2)
            poly = PolynomialFeatures(degree=2, include_bias=False)
            benchmark_operation(
                "PolynomialFeatures",
                lambda: poly.fit_transform(data),
                data
            )
            
            # Power transformation
            pt = PowerTransformer(method='yeo-johnson')
            pt.fit(data)
            benchmark_operation(
                "PowerTransform",
                lambda: pt.transform(data),
                data
            )
            
            # Binarization
            binarizer = Binarizer(threshold=0.0)
            benchmark_operation(
                "Binarize",
                lambda: binarizer.transform(data),
                data
            )


def benchmark_dimensionality_reduction():
    """Benchmark dimensionality reduction operations."""
    print("\n=== Dimensionality Reduction Benchmarks ===")
    
    for n_samples in [100, 500]:
        for n_features in [20, 50]:
            print(f"\nData shape: {n_samples}x{n_features}")
            data = np.random.uniform(-10, 10, (n_samples, n_features))
            n_components = n_features // 2
            
            # PCA
            pca = PCA(n_components=n_components)
            pca.fit(data)
            benchmark_operation(
                "PCA",
                lambda: pca.transform(data),
                data
            )
            
            # TruncatedSVD
            svd = TruncatedSVD(n_components=n_components)
            svd.fit(data)
            benchmark_operation(
                "TruncatedSVD",
                lambda: svd.transform(data),
                data
            )


def benchmark_imputation():
    """Benchmark imputation operations."""
    print("\n=== Imputation Benchmarks ===")
    
    for n_samples in SAMPLE_SIZES:
        for n_features in [10, 20]:
            print(f"\nData shape: {n_samples}x{n_features}")
            # Create data with missing values
            data = np.random.uniform(-10, 10, (n_samples, n_features))
            mask = np.random.random((n_samples, n_features)) < 0.1
            data[mask] = np.nan
            
            # SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(data)
            benchmark_operation(
                "SimpleImputer",
                lambda: imputer.transform(data),
                data
            )
            
            # KNNImputer (only for smaller datasets)
            if n_samples <= 1000:
                imputer = KNNImputer(n_neighbors=5)
                imputer.fit(data)
                benchmark_operation(
                    "KNNImputer",
                    lambda: imputer.transform(data),
                    data
                )


def benchmark_pipeline():
    """Benchmark pipeline operations."""
    print("\n=== Pipeline Benchmarks ===")
    
    for n_samples in [100, 1000]:
        for n_features in [10, 20]:
            print(f"\nData shape: {n_samples}x{n_features}")
            data = np.random.uniform(-10, 10, (n_samples, n_features))
            
            # Simple pipeline: StandardScaler -> PCA
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_features // 2))
            ])
            pipeline.fit(data)
            
            benchmark_operation(
                "StandardScaler_PCA",
                lambda: pipeline.transform(data),
                data
            )


def main():
    """Run all benchmarks."""
    print("Scikit-learn Transformation Benchmarks")
    print("=====================================")
    print(f"NumPy version: {np.__version__}")
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
    
    benchmark_normalization()
    benchmark_scaling()
    benchmark_feature_engineering()
    benchmark_dimensionality_reduction()
    benchmark_imputation()
    benchmark_pipeline()
    
    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()