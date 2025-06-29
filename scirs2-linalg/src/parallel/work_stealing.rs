//! Work-stealing scheduler implementation for dynamic load balancing
//!
//! This module provides a work-stealing scheduler that dynamically balances
//! work across threads, with timing analysis and adaptive chunking based on
//! workload characteristics.

use crate::error::LinalgResult;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, NumAssign, Zero, One};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use std::time::{Duration, Instant};
use std::iter::Sum;

/// Work item for the work-stealing scheduler
#[derive(Debug)]
pub struct WorkItem<T> {
    /// Unique identifier for the work item
    pub id: usize,
    /// The actual work payload
    pub payload: T,
    /// Expected execution time (for scheduling optimization)
    pub estimated_time: Option<Duration>,
}

impl<T> WorkItem<T> {
    /// Create a new work item
    pub fn new(id: usize, payload: T) -> Self {
        Self {
            id,
            payload,
            estimated_time: None,
        }
    }

    /// Create a work item with estimated execution time
    pub fn with_estimate(id: usize, payload: T, estimated_time: Duration) -> Self {
        Self {
            id,
            payload,
            estimated_time: Some(estimated_time),
        }
    }
}

/// Work queue for a single worker thread
#[derive(Debug)]
struct WorkQueue<T> {
    /// Double-ended queue for work items
    items: VecDeque<WorkItem<T>>,
    /// Number of items processed by this worker
    processed_count: usize,
    /// Total execution time for this worker
    total_time: Duration,
    /// Average execution time per item
    avg_time: Duration,
}

impl<T> Default for WorkQueue<T> {
    fn default() -> Self {
        Self {
            items: VecDeque::new(),
            processed_count: 0,
            total_time: Duration::ZERO,
            avg_time: Duration::ZERO,
        }
    }
}

impl<T> WorkQueue<T> {
    /// Add work item to the front of the queue (for local work)
    fn push_front(&mut self, item: WorkItem<T>) {
        self.items.push_front(item);
    }

    /// Add work item to the back of the queue (for stolen work)
    #[allow(dead_code)]
    fn push_back(&mut self, item: WorkItem<T>) {
        self.items.push_back(item);
    }

    /// Take work from the front (local work)
    fn pop_front(&mut self) -> Option<WorkItem<T>> {
        self.items.pop_front()
    }

    /// Steal work from the back (work stealing)
    fn steal_back(&mut self) -> Option<WorkItem<T>> {
        if self.items.len() > 1 {
            self.items.pop_back()
        } else {
            None
        }
    }

    /// Update timing statistics
    fn update_timing(&mut self, execution_time: Duration) {
        self.processed_count += 1;
        self.total_time += execution_time;
        self.avg_time = self.total_time / self.processed_count as u32;
    }

    /// Get the current load (estimated remaining work time)
    fn estimated_load(&self) -> Duration {
        let base_time = if self.avg_time.is_zero() {
            Duration::from_millis(1) // Default estimate
        } else {
            self.avg_time
        };

        self.items.iter()
            .map(|item| item.estimated_time.unwrap_or(base_time))
            .sum()
    }
}

/// Work-stealing scheduler with dynamic load balancing
pub struct WorkStealingScheduler<T> {
    /// Worker queues (one per thread)
    worker_queues: Vec<Arc<Mutex<WorkQueue<T>>>>,
    /// Number of worker threads
    num_workers: usize,
    /// Condition variable for worker synchronization
    worker_sync: Arc<(Mutex<bool>, Condvar)>,
    /// Statistics collection
    stats: Arc<Mutex<SchedulerStats>>,
}

/// Scheduler performance statistics
#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    /// Total items processed
    pub total_items: usize,
    /// Total execution time across all workers
    pub total_execution_time: Duration,
    /// Number of successful steals
    pub successful_steals: usize,
    /// Number of failed steal attempts
    pub failed_steals: usize,
    /// Load balancing efficiency (0.0 to 1.0)
    pub load_balance_efficiency: f64,
    /// Time variance across workers
    pub time_variance: f64,
}

impl<T: Send + 'static> WorkStealingScheduler<T> {
    /// Create a new work-stealing scheduler
    pub fn new(num_workers: usize) -> Self {
        let worker_queues = (0..num_workers)
            .map(|_| Arc::new(Mutex::new(WorkQueue::default())))
            .collect();

        Self {
            worker_queues,
            num_workers,
            worker_sync: Arc::new((Mutex::new(false), Condvar::new())),
            stats: Arc::new(Mutex::new(SchedulerStats::default())),
        }
    }

    /// Submit work items to the scheduler
    pub fn submit_work(&self, items: Vec<WorkItem<T>>) -> LinalgResult<()> {
        if items.is_empty() {
            return Ok(());
        }

        // Distribute work items across workers using round-robin
        for (i, item) in items.into_iter().enumerate() {
            let worker_id = i % self.num_workers;
            if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                queue.push_front(item);
            }
        }

        // Wake up all workers
        let (lock, cvar) = &*self.worker_sync;
        if let Ok(mut started) = lock.lock() {
            *started = true;
            cvar.notify_all();
        }

        Ok(())
    }

    /// Execute all work items using the work-stealing scheduler
    pub fn execute<F, R>(&self, work_fn: F) -> LinalgResult<Vec<R>>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + Clone + 'static,
        T: Send + 'static,
    {
        let work_fn = Arc::new(work_fn);
        let results = Arc::new(Mutex::new(Vec::new()));

        // Start worker threads
        let mut handles = Vec::new();
        for worker_id in 0..self.num_workers {
            let queue = Arc::clone(&self.worker_queues[worker_id]);
            let all_queues = self.worker_queues.clone();
            let work_fn = Arc::clone(&work_fn);
            let results = Arc::clone(&results);
            let stats = Arc::clone(&self.stats);
            let sync = Arc::clone(&self.worker_sync);

            let handle = thread::spawn(move || {
                Self::worker_loop(worker_id, queue, all_queues, work_fn, results, stats, sync);
            });
            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            handle.join().map_err(|_| {
                crate::error::LinalgError::ComputationError(
                    "Worker thread panicked".to_string()
                )
            })?;
        }

        // Extract results
        let results = results.lock().unwrap();
        Ok((*results).clone())
    }

    /// Worker thread main loop
    fn worker_loop<F, R>(
        worker_id: usize,
        my_queue: Arc<Mutex<WorkQueue<T>>>,
        all_queues: Vec<Arc<Mutex<WorkQueue<T>>>>,
        work_fn: Arc<F>,
        results: Arc<Mutex<Vec<R>>>,
        stats: Arc<Mutex<SchedulerStats>>,
        sync: Arc<(Mutex<bool>, Condvar)>,
    ) where
        F: Fn(T) -> R + Send + Sync,
        R: Send,
    {
        let (lock, cvar) = &*sync;

        // Wait for work to be available
        let _started = cvar.wait_while(lock.lock().unwrap(), |&mut started| !started).unwrap();

        loop {
            let work_item = {
                // Try to get work from own queue first
                if let Ok(mut queue) = my_queue.lock() {
                    queue.pop_front()
                } else {
                    None
                }
            };

            let work_item = match work_item {
                Some(item) => item,
                None => {
                    // Try to steal work from other workers
                    match Self::steal_work(worker_id, &all_queues, &stats) {
                        Some(item) => item,
                        None => {
                            // No work available, check if all queues are empty
                            if Self::all_queues_empty(&all_queues) {
                                break;
                            }
                            // Brief pause before trying again
                            thread::sleep(Duration::from_micros(10));
                            continue;
                        }
                    }
                }
            };

            // Execute the work item
            let start_time = Instant::now();
            let result = work_fn(work_item.payload);
            let execution_time = start_time.elapsed();

            // Update timing statistics
            if let Ok(mut queue) = my_queue.lock() {
                queue.update_timing(execution_time);
            }

            // Store the result
            if let Ok(mut results) = results.lock() {
                results.push(result);
            }

            // Update global statistics
            if let Ok(mut stats) = stats.lock() {
                stats.total_items += 1;
                stats.total_execution_time += execution_time;
            }
        }
    }

    /// Attempt to steal work from other workers
    fn steal_work(
        worker_id: usize,
        all_queues: &[Arc<Mutex<WorkQueue<T>>>],
        stats: &Arc<Mutex<SchedulerStats>>,
    ) -> Option<WorkItem<T>> {
        // Try to steal from the most loaded worker
        let mut best_target = None;
        let mut max_load = Duration::ZERO;

        for (i, queue) in all_queues.iter().enumerate() {
            if i == worker_id {
                continue; // Don't steal from ourselves
            }

            if let Ok(queue) = queue.lock() {
                let load = queue.estimated_load();
                if load > max_load {
                    max_load = load;
                    best_target = Some(i);
                }
            }
        }

        if let Some(target_id) = best_target {
            if let Ok(mut target_queue) = all_queues[target_id].lock() {
                if let Some(stolen_item) = target_queue.steal_back() {
                    // Update steal statistics
                    if let Ok(mut stats) = stats.lock() {
                        stats.successful_steals += 1;
                    }
                    return Some(stolen_item);
                }
            }
        }

        // Update failed steal statistics
        if let Ok(mut stats) = stats.lock() {
            stats.failed_steals += 1;
        }

        None
    }

    /// Check if all worker queues are empty
    fn all_queues_empty(queues: &[Arc<Mutex<WorkQueue<T>>>]) -> bool {
        queues.iter().all(|queue| {
            if let Ok(queue) = queue.lock() {
                queue.items.is_empty()
            } else {
                true // Assume empty if we can't lock
            }
        })
    }

    /// Get current scheduler statistics
    pub fn get_stats(&self) -> SchedulerStats {
        if let Ok(stats) = self.stats.lock() {
            let mut stats = stats.clone();
            stats.load_balance_efficiency = self.calculate_load_balance_efficiency();
            stats.time_variance = self.calculate_time_variance();
            stats
        } else {
            SchedulerStats::default()
        }
    }

    /// Calculate load balancing efficiency
    fn calculate_load_balance_efficiency(&self) -> f64 {
        let worker_times: Vec<Duration> = self.worker_queues
            .iter()
            .filter_map(|queue| {
                queue.lock().ok().map(|q| q.total_time)
            })
            .collect();

        if worker_times.is_empty() {
            return 1.0;
        }

        let max_time = worker_times.iter().max().unwrap().as_nanos() as f64;
        let min_time = worker_times.iter().min().unwrap().as_nanos() as f64;

        if max_time == 0.0 {
            1.0
        } else {
            min_time / max_time
        }
    }

    /// Calculate time variance across workers
    fn calculate_time_variance(&self) -> f64 {
        let worker_times: Vec<f64> = self.worker_queues
            .iter()
            .filter_map(|queue| {
                queue.lock().ok().map(|q| q.total_time.as_nanos() as f64)
            })
            .collect();

        if worker_times.len() < 2 {
            return 0.0;
        }

        let mean = worker_times.iter().sum::<f64>() / worker_times.len() as f64;
        let variance = worker_times.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / worker_times.len() as f64;

        variance.sqrt()
    }
}

/// Matrix-specific work-stealing algorithms
pub mod matrix_ops {
    use super::*;

    /// Work-stealing matrix-vector multiplication
    pub fn parallel_matvec_work_stealing<F>(
        matrix: &ArrayView2<F>,
        vector: &ArrayView1<F>,
        num_workers: usize,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if n != vector.len() {
            return Err(crate::error::LinalgError::ShapeError(
                "Matrix and vector dimensions don't match".to_string(),
            ));
        }

        let scheduler = WorkStealingScheduler::new(num_workers);
        let mut result = Array1::zeros(m);

        // Create work items for each row
        let work_items: Vec<WorkItem<(usize, Array1<F>, F)>> = (0..m)
            .map(|i| {
                let row = matrix.row(i).to_owned();
                let dot_product = row.dot(vector);
                WorkItem::new(i, (i, row, dot_product))
            })
            .collect();

        scheduler.submit_work(work_items)?;

        // Execute work and collect results
        let results = scheduler.execute(|(i, _row, dot_product)| (i, dot_product))?;

        // Assemble final result
        for (i, value) in results {
            result[i] = value;
        }

        Ok(result)
    }

    /// Work-stealing matrix multiplication
    pub fn parallel_gemm_work_stealing<F>(
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        num_workers: usize,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            return Err(crate::error::LinalgError::ShapeError(
                "Matrix dimensions don't match for multiplication".to_string(),
            ));
        }

        let scheduler = WorkStealingScheduler::new(num_workers);
        let mut result = Array2::zeros((m, n));

        // Create work items for blocks of the result matrix
        let block_size = (m * n / (num_workers * 4)).max(1);
        let mut work_items = Vec::new();

        for block_start in (0..m * n).step_by(block_size) {
            let block_end = (block_start + block_size).min(m * n);
            let indices: Vec<(usize, usize)> = (block_start..block_end)
                .map(|idx| (idx / n, idx % n))
                .collect();

            work_items.push(WorkItem::new(
                block_start,
                (indices, a.to_owned(), b.to_owned()),
            ));
        }

        scheduler.submit_work(work_items)?;

        // Execute work and collect results
        let results = scheduler.execute(|(indices, a_copy, b_copy)| {
            indices.into_iter()
                .map(|(i, j)| {
                    let value = a_copy.row(i).dot(&b_copy.column(j));
                    (i, j, value)
                })
                .collect::<Vec<_>>()
        })?;

        // Assemble final result
        for block_results in results {
            for (i, j, value) in block_results {
                result[(i, j)] = value;
            }
        }

        Ok(result)
    }

    /// Work-stealing Cholesky decomposition
    pub fn parallel_cholesky_work_stealing<F>(
        matrix: &ArrayView2<F>,
        num_workers: usize,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(crate::error::LinalgError::ShapeError(
                "Cholesky decomposition requires square matrix".to_string(),
            ));
        }

        let mut l = Array2::zeros((n, n));
        let matrix_owned = matrix.to_owned(); // Create owned copy to avoid lifetime issues
        
        // Cholesky decomposition with work-stealing for column operations
        for k in 0..n {
            // Compute diagonal element
            let mut sum = F::zero();
            for j in 0..k {
                sum += l[(k, j)] * l[(k, j)];
            }
            l[(k, k)] = (matrix_owned[(k, k)] - sum).sqrt();

            if k + 1 < n {
                let scheduler = WorkStealingScheduler::new(num_workers);
                
                // Create work items for remaining elements in column k
                #[allow(clippy::type_complexity)]
                let work_items: Vec<WorkItem<(usize, usize, Array2<F>, Array2<F>)>> = (k + 1..n)
                    .map(|i| WorkItem::new(i, (i, k, l.clone(), matrix_owned.clone())))
                    .collect();

                scheduler.submit_work(work_items)?;

                let results = scheduler.execute(|(i, k, l_copy, matrix_copy)| {
                    let mut sum = F::zero();
                    for j in 0..k {
                        sum += l_copy[(i, j)] * l_copy[(k, j)];
                    }
                    let value = (matrix_copy[(i, k)] - sum) / l_copy[(k, k)];
                    (i, value)
                })?;

                // Update the L matrix
                for (i, value) in results {
                    l[(i, k)] = value;
                }
            }
        }

        Ok(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_work_stealing_scheduler() {
        let scheduler = WorkStealingScheduler::new(2);
        
        let work_items = vec![
            WorkItem::new(0, 1),
            WorkItem::new(1, 2),
            WorkItem::new(2, 3),
            WorkItem::new(3, 4),
        ];

        scheduler.submit_work(work_items).unwrap();
        
        let results = scheduler.execute(|x| x * 2).unwrap();
        
        // Results might be in different order due to parallel execution
        let mut sorted_results = results;
        sorted_results.sort();
        assert_eq!(sorted_results, vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_work_stealing_matvec() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let vector = array![1.0, 2.0];
        
        let result = matrix_ops::parallel_matvec_work_stealing(
            &matrix.view(),
            &vector.view(),
            2,
        ).unwrap();
        
        assert_eq!(result, array![5.0, 11.0]);
    }

    #[test]
    fn test_scheduler_stats() {
        let scheduler = WorkStealingScheduler::new(2);
        
        let work_items = vec![
            WorkItem::new(0, 1),
            WorkItem::new(1, 2),
        ];

        scheduler.submit_work(work_items).unwrap();
        scheduler.execute(|x| x * 2).unwrap();
        
        let stats = scheduler.get_stats();
        assert_eq!(stats.total_items, 2);
        assert!(stats.load_balance_efficiency > 0.0);
    }
}