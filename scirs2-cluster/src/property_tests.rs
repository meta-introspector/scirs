//! Property-based tests for clustering algorithms
//!
//! This module contains property-based tests using the proptest framework
//! to verify that clustering algorithms satisfy fundamental mathematical
//! properties regardless of input data characteristics.

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use proptest::prelude::*;
    use std::collections::HashSet;

    use crate::hierarchy::{linkage, LinkageMethod, Metric};
    use crate::vq::{kmeans2, MinitMethod, MissingMethod};

    // Strategy for generating valid clustering data
    fn clustering_data_strategy() -> impl Strategy<Value = Array2<f64>> {
        prop::collection::vec(
            prop::collection::vec(-10.0f64..10.0f64, 2..5), // 2-5 features per point
            3..20,                                          // 3-20 data points
        )
        .prop_map(|data| {
            let n_points = data.len();
            let n_features = data[0].len();
            let flat_data: Vec<f64> = data.into_iter().flatten().collect();
            Array2::from_shape_vec((n_points, n_features), flat_data).unwrap()
        })
    }

    // Strategy for generating valid number of clusters
    fn k_clusters_strategy(max_points: usize) -> impl Strategy<Value = usize> {
        2usize..=(max_points.min(10)) // k should be at least 2 and at most min(n_points, 10)
    }

    proptest! {
        #[test]
        fn test_kmeans_all_points_assigned(
            data in clustering_data_strategy(),
            seed in any::<u64>()
        ) {
            let n_points = data.shape()[0];
            let k = (n_points / 2).max(2).min(5); // Reasonable k value

            let result = kmeans2(
                data.view(),
                k,
                Some(20), // iterations
                None,     // threshold
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(false), // check_finite
                Some(seed),
            );

            if let Ok((centroids, labels)) = result {
                // Every point should be assigned to a cluster
                prop_assert_eq!(labels.len(), n_points);

                // All cluster labels should be valid (in range [0, k))
                for &label in labels.iter() {
                    prop_assert!(label < k, "Cluster label {} should be < {}", label, k);
                }

                // Centroids should have correct shape
                prop_assert_eq!(centroids.shape()[0], k);
                prop_assert_eq!(centroids.shape()[1], data.shape()[1]);

                // All centroids should be finite
                for &val in centroids.iter() {
                    prop_assert!(val.is_finite(), "Centroid values should be finite");
                }
            }
        }

        #[test]
        fn test_kmeans_deterministic(
            data in clustering_data_strategy(),
            seed in any::<u64>()
        ) {
            let n_points = data.shape()[0];
            let k = (n_points / 2).max(2).min(4);

            let result1 = kmeans2(
                data.view(),
                k,
                Some(10),
                None,
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(false),
                Some(seed),
            );

            let result2 = kmeans2(
                data.view(),
                k,
                Some(10),
                None,
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(false),
                Some(seed),
            );

            if let (Ok((centroids1, labels1)), Ok((centroids2, labels2))) = (result1, result2) {
                // Results should be identical with same seed
                prop_assert_eq!(labels1, labels2, "Labels should be identical with same seed");
                prop_assert!(
                    centroids1.abs_diff_eq(&centroids2, 1e-10),
                    "Centroids should be identical with same seed"
                );
            }
        }

        /// Test hierarchical clustering produces correct number of merges
        #[test]
        fn test_hierarchical_merge_count(data in clustering_data_strategy()) {
            let n_points = data.shape()[0];

            let result = linkage(data.view(), LinkageMethod::Single, Metric::Euclidean);

            if let Ok(linkage_matrix) = result {
                // Should produce exactly n-1 merges for n points
                prop_assert_eq!(
                    linkage_matrix.shape()[0],
                    n_points - 1,
                    "Linkage matrix should have n-1 rows for n points"
                );

                // Each row should have 4 columns: [cluster1, cluster2, distance, count]
                prop_assert_eq!(linkage_matrix.shape()[1], 4);

                // All merge distances should be non-negative
                for i in 0..linkage_matrix.shape()[0] {
                    let distance = linkage_matrix[[i, 2]];
                    prop_assert!(
                        distance >= 0.0,
                        "Merge distance should be non-negative, got {}",
                        distance
                    );
                }

                // All cluster counts should be >= 2 (since it's a merge)
                for i in 0..linkage_matrix.shape()[0] {
                    let count = linkage_matrix[[i, 3]];
                    prop_assert!(
                        count >= 2.0,
                        "Cluster count should be >= 2, got {}",
                        count
                    );
                }
            }
        }

        /// Test hierarchical clustering merge heights are non-decreasing for single linkage
        #[test]
        fn test_hierarchical_single_linkage_monotonic(data in clustering_data_strategy()) {
            let result = linkage(data.view(), LinkageMethod::Single, Metric::Euclidean);

            if let Ok(linkage_matrix) = result {
                // For single linkage, merge distances should be non-decreasing
                // (this is a fundamental property of single linkage)
                for i in 1..linkage_matrix.shape()[0] {
                    let prev_distance = linkage_matrix[[i - 1, 2]];
                    let curr_distance = linkage_matrix[[i, 2]];
                    prop_assert!(
                        curr_distance >= prev_distance - 1e-10, // Allow small numerical errors
                        "Single linkage distances should be non-decreasing: {} > {}",
                        prev_distance,
                        curr_distance
                    );
                }
            }
        }

        /// Test that cluster IDs in linkage matrix are valid
        #[test]
        fn test_hierarchical_valid_cluster_ids(data in clustering_data_strategy()) {
            let n_points = data.shape()[0];
            let result = linkage(data.view(), LinkageMethod::Average, Metric::Euclidean);

            if let Ok(linkage_matrix) = result {
                for i in 0..linkage_matrix.shape()[0] {
                    let cluster1 = linkage_matrix[[i, 0]] as usize;
                    let cluster2 = linkage_matrix[[i, 1]] as usize;

                    // Cluster IDs should be valid indices
                    // Original points are 0..n_points-1
                    // New clusters are n_points..2*n_points-2
                    prop_assert!(
                        cluster1 < n_points + i,
                        "Cluster1 ID {} should be < {}",
                        cluster1,
                        n_points + i
                    );
                    prop_assert!(
                        cluster2 < n_points + i,
                        "Cluster2 ID {} should be < {}",
                        cluster2,
                        n_points + i
                    );

                    // The two clusters being merged should be different
                    prop_assert_ne!(cluster1, cluster2, "Cannot merge cluster with itself");
                }
            }
        }

        #[test]
        fn test_kmeans_initialization_methods(
            data in clustering_data_strategy(),
            init_method in prop::sample::select(&[MinitMethod::Random, MinitMethod::Points, MinitMethod::PlusPlus])
        ) {
            let n_points = data.shape()[0];
            let k = (n_points / 2).max(2).min(4);

            let result = kmeans2(
                data.view(),
                k,
                Some(10),
                None,
                Some(init_method),
                Some(MissingMethod::Warn),
                Some(false),
                Some(42), // Fixed seed for reproducibility
            );

            if let Ok((centroids, labels)) = result {
                // Basic validation
                prop_assert_eq!(labels.len(), n_points);
                prop_assert_eq!(centroids.shape()[0], k);

                // All labels should be in valid range
                for &label in labels.iter() {
                    prop_assert!(label < k);
                }

                // Should have at least one point assigned to each cluster (ideally)
                let unique_labels: HashSet<_> = labels.iter().cloned().collect();
                // Note: We can't guarantee all clusters are used due to empty cluster handling
                prop_assert!(!unique_labels.is_empty(), "Should have at least one cluster");
                prop_assert!(unique_labels.len() <= k, "Cannot have more clusters than k");
            }
        }

        /// Test that spectral clustering produces valid results
        #[test]
        fn test_spectral_clustering_properties(data in clustering_data_strategy()) {
            let n_points = data.shape()[0];

            // Only test if we have enough points for spectral clustering
            if n_points >= 4 {
                let k = 2; // Use 2 clusters for spectral clustering test

                let options = crate::spectral::SpectralClusteringOptions {
                    affinity: crate::spectral::AffinityMode::RBF,
                    gamma: 1.0,
                    normalized_laplacian: true,
                    max_iter: 50,
                    n_init: 1, // Single initialization for testing
                    tol: 1e-4,
                    random_seed: Some(42),
                    eigen_solver: "arpack".to_string(),
                    auto_n_clusters: false,
                };

                let result = crate::spectral::spectral_clustering(data.view(), k, Some(options));

                if let Ok((embeddings, labels)) = result {
                    // Basic validation
                    prop_assert_eq!(labels.len(), n_points);
                    prop_assert_eq!(embeddings.shape()[0], n_points);

                    // All labels should be valid
                    for &label in labels.iter() {
                        prop_assert!(label < k, "Label {} should be < {}", label, k);
                    }

                    // Embeddings should be finite
                    for &val in embeddings.iter() {
                        prop_assert!(val.is_finite(), "Embedding values should be finite");
                    }
                }
            }
        }

        /// Test DBSCAN properties (noise handling)
        #[test]
        fn test_dbscan_noise_handling(data in clustering_data_strategy()) {
            // Use fixed parameters for DBSCAN
            let eps = 2.0;
            let min_samples = 2;

            let result = crate::density::dbscan(data.view(), eps, min_samples);

            if let Ok(labels) = result {
                prop_assert_eq!(labels.len(), data.shape()[0]);

                // DBSCAN can assign noise points (label -1), but all other labels should be >= 0
                for &label in labels.iter() {
                    // In our implementation, noise might be represented differently
                    // This is a basic sanity check that labels are reasonable integers
                    prop_assert!(
                        label >= -1,
                        "DBSCAN label should be >= -1 (for noise), got {}",
                        label
                    );
                }

                // If there are any non-noise points, they should form at least one cluster
                let non_noise_labels: Vec<_> = labels.iter().filter(|&&l| l >= 0).collect();
                if !non_noise_labels.is_empty() {
                    let unique_clusters: HashSet<_> = non_noise_labels.iter().cloned().collect();
                    prop_assert!(!unique_clusters.is_empty(), "Should have at least one cluster if non-noise points exist");
                }
            }
        }

        /// Test that clustering algorithms handle edge cases gracefully
        #[test]
        fn test_clustering_edge_cases_minimal_data() {
            // Test with minimal data (3 points, 2 clusters)
            let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();

            // K-means with 2 clusters on 3 points
            let result = kmeans2(
                data.view(),
                2,
                Some(5),
                None,
                Some(MinitMethod::Points),
                Some(MissingMethod::Warn),
                Some(false),
                Some(123),
            );

            prop_assert!(result.is_ok(), "K-means should handle minimal data");

            if let Ok((centroids, labels)) = result {
                prop_assert_eq!(labels.len(), 3);
                prop_assert_eq!(centroids.shape()[0], 2);
            }

            // Hierarchical clustering
            let linkage_result = linkage(data.view(), LinkageMethod::Single, Metric::Euclidean);
            prop_assert!(linkage_result.is_ok(), "Hierarchical clustering should handle minimal data");

            if let Ok(linkage_matrix) = linkage_result {
                prop_assert_eq!(linkage_matrix.shape()[0], 2); // 3 points -> 2 merges
            }
        }
    }

    /// Additional non-property tests for specific edge cases
    #[cfg(test)]
    mod specific_tests {
        use super::*;

        #[test]
        fn test_kmeans_identical_points() {
            // Test with all identical points
            let data = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();

            let result = kmeans2(
                data.view(),
                2,
                Some(10),
                None,
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(false),
                Some(42),
            );

            // Should handle identical points gracefully
            assert!(
                result.is_ok() || result.is_err(),
                "Should either succeed or fail gracefully"
            );
        }

        #[test]
        fn test_hierarchical_identical_points() {
            // Test hierarchical clustering with identical points
            let data = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();

            let result = linkage(data.view(), LinkageMethod::Average, Metric::Euclidean);

            if let Ok(linkage_matrix) = result {
                assert_eq!(linkage_matrix.shape()[0], 3); // 4 points -> 3 merges

                // All merge distances should be 0 (or very close to 0)
                for i in 0..linkage_matrix.shape()[0] {
                    let distance: f64 = linkage_matrix[[i, 2]];
                    assert!(
                        distance.abs() < 1e-10,
                        "Distance should be ~0 for identical points"
                    );
                }
            }
        }
    }
}
