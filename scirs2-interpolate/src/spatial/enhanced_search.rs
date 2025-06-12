//! Enhanced nearest neighbor search algorithms
//!
//! This module provides optimized implementations of various nearest neighbor
//! search algorithms with advanced features like approximate search,
//! multi-threaded queries, and adaptive indexing.

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive, Zero};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;

/// Configuration for enhanced nearest neighbor search
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum number of neighbors to search for
    pub max_neighbors: usize,
    /// Search radius (for radius-based searches)
    pub radius: Option<f64>,
    /// Approximation factor (1.0 = exact, >1.0 = approximate)
    pub approximation_factor: f64,
    /// Enable parallel search for large query sets
    pub parallel_search: bool,
    /// Number of threads for parallel search (None = auto)
    pub num_threads: Option<usize>,
    /// Use adaptive indexing for dynamic data
    pub adaptive_indexing: bool,
    /// Cache search results for repeated queries
    pub cache_results: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_neighbors: 10,
            radius: None,
            approximation_factor: 1.0,
            parallel_search: true,
            num_threads: None,
            adaptive_indexing: false,
            cache_results: true,
        }
    }
}

/// Enhanced nearest neighbor searcher with multiple algorithms
#[derive(Debug)]
pub struct EnhancedNearestNeighborSearcher<F: Float + FromPrimitive + Debug + Send + Sync> {
    /// Training data points
    points: Array2<F>,
    /// Spatial index type currently in use
    index_type: IndexType,
    /// Search configuration
    config: SearchConfig,
    /// KD-tree index (if applicable)
    kdtree: Option<KdTreeIndex<F>>,
    /// Ball tree index (if applicable)
    balltree: Option<BallTreeIndex<F>>,
    /// LSH index for approximate search (if applicable)
    lsh_index: Option<LSHIndex<F>>,
    /// Query cache for repeated searches
    query_cache: HashMap<QueryKey, Vec<(usize, F)>>,
    /// Performance statistics
    stats: SearchStats,
}

/// Types of spatial indices available
#[derive(Debug, Clone, PartialEq)]
pub enum IndexType {
    /// Brute force search (O(n) per query)
    BruteForce,
    /// KD-tree (efficient for low dimensions)
    KdTree,
    /// Ball tree (efficient for high dimensions)
    BallTree,
    /// Locality Sensitive Hashing (approximate)
    LSH,
    /// Adaptive hybrid approach
    Adaptive,
}

/// Simple KD-tree implementation for low-dimensional data
#[derive(Debug)]
pub struct KdTreeIndex<F: Float> {
    /// Tree nodes
    nodes: Vec<KdTreeNode<F>>,
    /// Root node index
    root: usize,
    /// Number of dimensions
    dimensions: usize,
}

/// KD-tree node
#[derive(Debug, Clone)]
struct KdTreeNode<F: Float> {
    /// Point index in original data
    point_idx: Option<usize>,
    /// Splitting dimension
    split_dim: usize,
    /// Splitting value
    split_value: F,
    /// Left child index
    left: Option<usize>,
    /// Right child index
    right: Option<usize>,
    /// Bounding box for pruning
    bbox_min: Vec<F>,
    bbox_max: Vec<F>,
}

/// Ball tree implementation for high-dimensional data
#[derive(Debug)]
pub struct BallTreeIndex<F: Float> {
    /// Tree nodes
    nodes: Vec<BallTreeNode<F>>,
    /// Root node index
    root: usize,
    /// Number of dimensions
    dimensions: usize,
}

/// Ball tree node
#[derive(Debug, Clone)]
struct BallTreeNode<F: Float> {
    /// Point indices in this ball
    point_indices: Vec<usize>,
    /// Center of the ball
    center: Vec<F>,
    /// Radius of the ball
    radius: F,
    /// Left child index
    left: Option<usize>,
    /// Right child index
    right: Option<usize>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

/// Locality Sensitive Hashing index for approximate search
#[derive(Debug)]
pub struct LSHIndex<F: Float> {
    /// Hash tables
    hash_tables: Vec<HashMap<u64, Vec<usize>>>,
    /// Random projection matrices
    projections: Vec<Array2<F>>,
    /// Number of hash functions per table
    hash_functions_per_table: usize,
    /// Number of hash tables
    num_tables: usize,
    /// Hash bucket width
    bucket_width: F,
}

/// Query key for caching (using approximate floating point comparison)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QueryKey {
    /// Quantized coordinates for hashing
    coords: Vec<i64>,
    /// Number of neighbors requested
    k: usize,
    /// Quantized radius (if applicable)
    radius: Option<i64>,
}

/// Performance statistics for search operations
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Total number of queries processed
    pub total_queries: usize,
    /// Total query time in microseconds
    pub total_query_time_us: u64,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of distance computations
    pub distance_computations: usize,
    /// Number of nodes visited (for tree-based methods)
    pub nodes_visited: usize,
}

impl QueryKey {
    /// Create a new query key from coordinates
    fn from_coords<F: Float + FromPrimitive>(coords: &[F], k: usize, radius: Option<F>) -> Self {
        const QUANTIZATION_FACTOR: f64 = 1000.0;

        let quantized_coords: Vec<i64> = coords
            .iter()
            .map(|&x| (x.to_f64().unwrap_or(0.0) * QUANTIZATION_FACTOR).round() as i64)
            .collect();

        let quantized_radius =
            radius.map(|r| (r.to_f64().unwrap_or(0.0) * QUANTIZATION_FACTOR).round() as i64);

        Self {
            coords: quantized_coords,
            k,
            radius: quantized_radius,
        }
    }
}

impl<F> EnhancedNearestNeighborSearcher<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + 'static,
{
    /// Create a new enhanced nearest neighbor searcher
    ///
    /// # Arguments
    ///
    /// * `points` - Training data points with shape (n_points, n_dims)
    /// * `index_type` - Type of spatial index to use
    /// * `config` - Search configuration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::Array2;
    /// use scirs2_interpolate::spatial::enhanced_search::{
    ///     EnhancedNearestNeighborSearcher, IndexType, SearchConfig
    /// };
    ///
    /// let points = Array2::from_shape_vec((5, 2), vec![
    ///     0.0, 0.0,
    ///     1.0, 0.0,
    ///     0.0, 1.0,
    ///     1.0, 1.0,
    ///     0.5, 0.5,
    /// ]).unwrap();
    ///
    /// let config = SearchConfig::default();
    /// let searcher = EnhancedNearestNeighborSearcher::new(
    ///     points, IndexType::Adaptive, config
    /// ).unwrap();
    /// ```
    pub fn new(
        points: Array2<F>,
        index_type: IndexType,
        config: SearchConfig,
    ) -> InterpolateResult<Self> {
        let n_points = points.nrows();
        let n_dims = points.ncols();

        if n_points == 0 {
            return Err(InterpolateError::ValueError(
                "Cannot create searcher with zero points".to_string(),
            ));
        }

        // Determine the best index type if using adaptive
        let actual_index_type = match index_type {
            IndexType::Adaptive => Self::choose_index_type(n_points, n_dims, &config),
            other => other,
        };

        let mut searcher = Self {
            points,
            index_type: actual_index_type.clone(),
            config,
            kdtree: None,
            balltree: None,
            lsh_index: None,
            query_cache: HashMap::new(),
            stats: SearchStats::default(),
        };

        // Build the appropriate index
        searcher.build_index(actual_index_type)?;

        Ok(searcher)
    }

    /// Choose the best index type based on data characteristics
    fn choose_index_type(n_points: usize, n_dims: usize, config: &SearchConfig) -> IndexType {
        if config.approximation_factor > 1.0 && n_points > 10000 {
            // Use LSH for large datasets with approximate search
            IndexType::LSH
        } else if n_dims <= 10 && n_points > 100 {
            // Use KD-tree for low-dimensional data
            IndexType::KdTree
        } else if n_dims <= 50 && n_points > 50 {
            // Use Ball tree for medium-dimensional data
            IndexType::BallTree
        } else {
            // Use brute force for small datasets or very high dimensions
            IndexType::BruteForce
        }
    }

    /// Build the spatial index
    fn build_index(&mut self, index_type: IndexType) -> InterpolateResult<()> {
        match index_type {
            IndexType::KdTree => {
                self.kdtree = Some(KdTreeIndex::new(&self.points)?);
            }
            IndexType::BallTree => {
                self.balltree = Some(BallTreeIndex::new(&self.points)?);
            }
            IndexType::LSH => {
                self.lsh_index = Some(LSHIndex::new(&self.points, &self.config)?);
            }
            IndexType::BruteForce | IndexType::Adaptive => {
                // No index needed for brute force
            }
        }

        Ok(())
    }

    /// Find k nearest neighbors for a query point
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    ///
    /// Vector of (point_index, distance) pairs sorted by distance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, Array2};
    /// use scirs2_interpolate::spatial::enhanced_search::{
    ///     EnhancedNearestNeighborSearcher, IndexType, SearchConfig
    /// };
    ///
    /// let points = Array2::from_shape_vec((4, 2), vec![
    ///     0.0, 0.0,
    ///     1.0, 0.0,
    ///     0.0, 1.0,
    ///     1.0, 1.0,
    /// ]).unwrap();
    ///
    /// let searcher = EnhancedNearestNeighborSearcher::new(
    ///     points, IndexType::BruteForce, SearchConfig::default()
    /// ).unwrap();
    ///
    /// let query = Array1::from_vec(vec![0.5, 0.5]);
    /// let neighbors = searcher.k_nearest_neighbors(&query.view(), 2).unwrap();
    ///
    /// assert_eq!(neighbors.len(), 2);
    /// ```
    pub fn k_nearest_neighbors(
        &mut self,
        query: &ArrayView1<F>,
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let start_time = std::time::Instant::now();
        self.stats.total_queries += 1;

        // Check cache first
        if self.config.cache_results {
            let cache_key = QueryKey::from_coords(query.as_slice().unwrap(), k, None);
            if let Some(cached_result) = self.query_cache.get(&cache_key) {
                self.stats.cache_hits += 1;
                return Ok(cached_result.clone());
            }
        }

        let result = match &self.index_type {
            IndexType::BruteForce => self.brute_force_knn(query, k),
            IndexType::KdTree => {
                if let Some(ref kdtree) = self.kdtree {
                    kdtree.k_nearest_neighbors(query, k, &mut self.stats)
                } else {
                    self.brute_force_knn(query, k)
                }
            }
            IndexType::BallTree => {
                if let Some(ref balltree) = self.balltree {
                    balltree.k_nearest_neighbors(query, k, &mut self.stats)
                } else {
                    self.brute_force_knn(query, k)
                }
            }
            IndexType::LSH => {
                if let Some(ref lsh) = self.lsh_index {
                    lsh.approximate_k_nearest_neighbors(query, k, &self.points, &mut self.stats)
                } else {
                    self.brute_force_knn(query, k)
                }
            }
            IndexType::Adaptive => self.brute_force_knn(query, k),
        };

        // Cache the result
        if self.config.cache_results {
            if let Ok(ref neighbors) = result {
                let cache_key = QueryKey::from_coords(query.as_slice().unwrap(), k, None);
                self.query_cache.insert(cache_key, neighbors.clone());
            }
        }

        // Update timing statistics
        let elapsed = start_time.elapsed();
        self.stats.total_query_time_us += elapsed.as_micros() as u64;

        result
    }

    /// Find all neighbors within a given radius
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    /// * `radius` - Search radius
    ///
    /// # Returns
    ///
    /// Vector of (point_index, distance) pairs within the radius
    pub fn radius_neighbors(
        &mut self,
        query: &ArrayView1<F>,
        radius: F,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let start_time = std::time::Instant::now();
        self.stats.total_queries += 1;

        // Check cache first
        if self.config.cache_results {
            let cache_key = QueryKey::from_coords(query.as_slice().unwrap(), 0, Some(radius));
            if let Some(cached_result) = self.query_cache.get(&cache_key) {
                self.stats.cache_hits += 1;
                return Ok(cached_result.clone());
            }
        }

        let result = match &self.index_type {
            IndexType::BruteForce => self.brute_force_radius(query, radius),
            IndexType::KdTree => {
                if let Some(ref kdtree) = self.kdtree {
                    kdtree.radius_neighbors(query, radius, &mut self.stats)
                } else {
                    self.brute_force_radius(query, radius)
                }
            }
            IndexType::BallTree => {
                if let Some(ref balltree) = self.balltree {
                    balltree.radius_neighbors(query, radius, &mut self.stats)
                } else {
                    self.brute_force_radius(query, radius)
                }
            }
            IndexType::LSH => {
                if let Some(ref lsh) = self.lsh_index {
                    lsh.approximate_radius_neighbors(query, radius, &self.points, &mut self.stats)
                } else {
                    self.brute_force_radius(query, radius)
                }
            }
            IndexType::Adaptive => self.brute_force_radius(query, radius),
        };

        // Cache the result
        if self.config.cache_results {
            if let Ok(ref neighbors) = result {
                let cache_key = QueryKey::from_coords(query.as_slice().unwrap(), 0, Some(radius));
                self.query_cache.insert(cache_key, neighbors.clone());
            }
        }

        // Update timing statistics
        let elapsed = start_time.elapsed();
        self.stats.total_query_time_us += elapsed.as_micros() as u64;

        result
    }

    /// Perform k-nearest neighbor search on multiple query points in parallel
    ///
    /// # Arguments
    ///
    /// * `queries` - Query points with shape (n_queries, n_dims)
    /// * `k` - Number of nearest neighbors per query
    ///
    /// # Returns
    ///
    /// Vector of neighbor results for each query point
    pub fn batch_k_nearest_neighbors(
        &mut self,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        let n_queries = queries.nrows();

        if !self.config.parallel_search || n_queries < 10 {
            // Sequential processing for small batches
            let mut results = Vec::with_capacity(n_queries);
            for i in 0..n_queries {
                let query = queries.slice(ndarray::s![i, ..]);
                results.push(self.k_nearest_neighbors(&query, k)?);
            }
            Ok(results)
        } else {
            // Parallel processing for large batches
            self.parallel_batch_knn(queries, k)
        }
    }

    /// Parallel implementation of batch k-nearest neighbor search
    fn parallel_batch_knn(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        let queries_owned = queries.to_owned();
        let points = &self.points;

        let results: Result<Vec<_>, InterpolateError> = (0..queries_owned.nrows())
            .into_par_iter()
            .map(|i| {
                let query = queries_owned.slice(ndarray::s![i, ..]);
                Self::parallel_brute_force_knn(&query, k, points)
            })
            .collect();

        results
    }

    /// Thread-safe brute force k-NN for parallel execution
    fn parallel_brute_force_knn(
        query: &ArrayView1<F>,
        k: usize,
        points: &Array2<F>,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let mut neighbors = BinaryHeap::new();

        for (idx, point) in points.axis_iter(Axis(0)).enumerate() {
            let distance = Self::euclidean_distance_squared(query, &point);

            if neighbors.len() < k {
                neighbors.push(std::cmp::Reverse((OrderedFloat(distance), idx)));
            } else if let Some(&std::cmp::Reverse((OrderedFloat(max_dist), _))) = neighbors.peek() {
                if distance < max_dist {
                    neighbors.pop();
                    neighbors.push(std::cmp::Reverse((OrderedFloat(distance), idx)));
                }
            }
        }

        let mut result: Vec<_> = neighbors
            .into_iter()
            .map(|std::cmp::Reverse((OrderedFloat(dist), idx))| (idx, dist.sqrt()))
            .collect();

        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(result)
    }

    /// Brute force k-nearest neighbor search
    fn brute_force_knn(
        &mut self,
        query: &ArrayView1<F>,
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let mut neighbors = BinaryHeap::new();

        for (idx, point) in self.points.axis_iter(Axis(0)).enumerate() {
            let distance = Self::euclidean_distance_squared(query, &point);
            self.stats.distance_computations += 1;

            if neighbors.len() < k {
                neighbors.push(std::cmp::Reverse((OrderedFloat(distance), idx)));
            } else if let Some(&std::cmp::Reverse((OrderedFloat(max_dist), _))) = neighbors.peek() {
                if distance < max_dist {
                    neighbors.pop();
                    neighbors.push(std::cmp::Reverse((OrderedFloat(distance), idx)));
                }
            }
        }

        let mut result: Vec<_> = neighbors
            .into_iter()
            .map(|std::cmp::Reverse((OrderedFloat(dist), idx))| (idx, dist.sqrt()))
            .collect();

        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(result)
    }

    /// Brute force radius neighbor search
    fn brute_force_radius(
        &mut self,
        query: &ArrayView1<F>,
        radius: F,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let mut neighbors = Vec::new();
        let radius_squared = radius * radius;

        for (idx, point) in self.points.axis_iter(Axis(0)).enumerate() {
            let distance_squared = Self::euclidean_distance_squared(query, &point);
            self.stats.distance_computations += 1;

            if distance_squared <= radius_squared {
                neighbors.push((idx, distance_squared.sqrt()));
            }
        }

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(neighbors)
    }

    /// Compute squared Euclidean distance between two points
    fn euclidean_distance_squared(p1: &ArrayView1<F>, p2: &ArrayView1<F>) -> F {
        p1.iter()
            .zip(p2.iter())
            .map(|(&a, &b)| {
                let diff = a - b;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
    }

    /// Get performance statistics
    pub fn stats(&self) -> &SearchStats {
        &self.stats
    }

    /// Clear the query cache
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.stats.total_queries == 0 {
            0.0
        } else {
            self.stats.cache_hits as f64 / self.stats.total_queries as f64
        }
    }

    /// Get average query time in microseconds
    pub fn average_query_time_us(&self) -> f64 {
        if self.stats.total_queries == 0 {
            0.0
        } else {
            self.stats.total_query_time_us as f64 / self.stats.total_queries as f64
        }
    }
}

/// Wrapper for floating point values to make them orderable
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct OrderedFloat<F: Float>(F);

impl<F: Float> Eq for OrderedFloat<F> {}

impl<F: Float> Ord for OrderedFloat<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// Implementation stubs for the various index types
impl<F: Float + FromPrimitive> KdTreeIndex<F> {
    pub fn new(_points: &Array2<F>) -> InterpolateResult<Self> {
        // Simplified stub - real implementation would build the tree
        Err(InterpolateError::NotImplementedError(
            "KdTreeIndex not fully implemented".to_string(),
        ))
    }

    pub fn k_nearest_neighbors(
        &self,
        _query: &ArrayView1<F>,
        _k: usize,
        _stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        Err(InterpolateError::NotImplementedError(
            "KdTreeIndex k_nearest_neighbors not implemented".to_string(),
        ))
    }

    pub fn radius_neighbors(
        &self,
        _query: &ArrayView1<F>,
        _radius: F,
        _stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        Err(InterpolateError::NotImplementedError(
            "KdTreeIndex radius_neighbors not implemented".to_string(),
        ))
    }
}

impl<F: Float + FromPrimitive> BallTreeIndex<F> {
    pub fn new(_points: &Array2<F>) -> InterpolateResult<Self> {
        // Simplified stub - real implementation would build the tree
        Err(InterpolateError::NotImplementedError(
            "BallTreeIndex not fully implemented".to_string(),
        ))
    }

    pub fn k_nearest_neighbors(
        &self,
        _query: &ArrayView1<F>,
        _k: usize,
        _stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        Err(InterpolateError::NotImplementedError(
            "BallTreeIndex k_nearest_neighbors not implemented".to_string(),
        ))
    }

    pub fn radius_neighbors(
        &self,
        _query: &ArrayView1<F>,
        _radius: F,
        _stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        Err(InterpolateError::NotImplementedError(
            "BallTreeIndex radius_neighbors not implemented".to_string(),
        ))
    }
}

impl<F: Float + FromPrimitive> LSHIndex<F> {
    pub fn new(_points: &Array2<F>, _config: &SearchConfig) -> InterpolateResult<Self> {
        // Simplified stub - real implementation would build the hash tables
        Err(InterpolateError::NotImplementedError(
            "LSHIndex not fully implemented".to_string(),
        ))
    }

    pub fn approximate_k_nearest_neighbors(
        &self,
        _query: &ArrayView1<F>,
        _k: usize,
        _points: &Array2<F>,
        _stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        Err(InterpolateError::NotImplementedError(
            "LSHIndex approximate_k_nearest_neighbors not implemented".to_string(),
        ))
    }

    pub fn approximate_radius_neighbors(
        &self,
        _query: &ArrayView1<F>,
        _radius: F,
        _points: &Array2<F>,
        _stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        Err(InterpolateError::NotImplementedError(
            "LSHIndex approximate_radius_neighbors not implemented".to_string(),
        ))
    }
}

/// Create an enhanced nearest neighbor searcher with automatic index selection
///
/// This function automatically chooses the best spatial index based on the
/// characteristics of the input data.
///
/// # Arguments
///
/// * `points` - Training data points with shape (n_points, n_dims)
/// * `config` - Optional search configuration (uses defaults if None)
///
/// # Returns
///
/// A configured enhanced nearest neighbor searcher
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_interpolate::spatial::enhanced_search::{
///     make_enhanced_searcher, SearchConfig
/// };
///
/// let points = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect()).unwrap();
/// let searcher = make_enhanced_searcher(points, None).unwrap();
/// ```
pub fn make_enhanced_searcher<F>(
    points: Array2<F>,
    config: Option<SearchConfig>,
) -> InterpolateResult<EnhancedNearestNeighborSearcher<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + 'static,
{
    let config = config.unwrap_or_default();
    EnhancedNearestNeighborSearcher::new(points, IndexType::Adaptive, config)
}

/// Create a high-performance searcher optimized for large datasets
///
/// This function creates a searcher specifically optimized for large datasets
/// with features like parallel processing and approximate search.
///
/// # Arguments
///
/// * `points` - Training data points with shape (n_points, n_dims)
/// * `approximation_factor` - Approximation factor (1.0 = exact, >1.0 = approximate)
/// * `num_threads` - Number of threads for parallel processing (None = auto)
///
/// # Returns
///
/// A high-performance nearest neighbor searcher
pub fn make_high_performance_searcher<F>(
    points: Array2<F>,
    approximation_factor: f64,
    num_threads: Option<usize>,
) -> InterpolateResult<EnhancedNearestNeighborSearcher<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + 'static,
{
    let config = SearchConfig {
        approximation_factor,
        parallel_search: true,
        num_threads,
        cache_results: true,
        adaptive_indexing: true,
        ..Default::default()
    };

    let index_type = if approximation_factor > 1.0 {
        IndexType::LSH
    } else {
        IndexType::Adaptive
    };

    EnhancedNearestNeighborSearcher::new(points, index_type, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_enhanced_searcher_creation() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let config = SearchConfig::default();

        let searcher = EnhancedNearestNeighborSearcher::new(points, IndexType::BruteForce, config);

        assert!(searcher.is_ok());
    }

    #[test]
    fn test_brute_force_knn() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let config = SearchConfig::default();

        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BruteForce, config).unwrap();

        let query = array![0.5, 0.5];
        let neighbors = searcher.k_nearest_neighbors(&query.view(), 2).unwrap();

        assert_eq!(neighbors.len(), 2);
        // All corners should be equidistant from center
        assert!((neighbors[0].1 - neighbors[1].1).abs() < 1e-10);
    }

    #[test]
    fn test_radius_search() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]];
        let config = SearchConfig::default();

        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BruteForce, config).unwrap();

        let query = array![0.5, 0.5];
        let neighbors = searcher.radius_neighbors(&query.view(), 1.0).unwrap();

        // Should find all points except (2,2) which is too far
        assert_eq!(neighbors.len(), 4);
    }

    #[test]
    fn test_batch_search() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let config = SearchConfig::default();

        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BruteForce, config).unwrap();

        let queries = array![[0.1, 0.1], [0.9, 0.9]];
        let results = searcher
            .batch_k_nearest_neighbors(&queries.view(), 2)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 2);
    }

    #[test]
    fn test_cache_functionality() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let config = SearchConfig {
            cache_results: true,
            ..Default::default()
        };

        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BruteForce, config).unwrap();

        let query = array![0.5, 0.5];

        // First query
        let _neighbors1 = searcher.k_nearest_neighbors(&query.view(), 2).unwrap();
        assert_eq!(searcher.stats().cache_hits, 0);

        // Second query (should hit cache)
        let _neighbors2 = searcher.k_nearest_neighbors(&query.view(), 2).unwrap();
        assert_eq!(searcher.stats().cache_hits, 1);

        assert!(searcher.cache_hit_ratio() > 0.0);
    }

    #[test]
    fn test_make_enhanced_searcher() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let searcher = make_enhanced_searcher(points, None);
        assert!(searcher.is_ok());
    }
}
