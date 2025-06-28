// Test to verify that the parallel_ops imports are working correctly
use scirs2_core::parallel_ops::{
    par_chunks, parallel_map, ParallelIterator, IntoParallelIterator, 
    ParallelBridge, num_threads
};

fn main() {
    println!("Testing parallel operations imports...");
    
    // Test 1: Number of threads
    let threads = num_threads();
    println!("Available threads: {}", threads);
    
    // Test 2: Parallel range iteration
    let data: Vec<i32> = (0..10).into_par_iter().map(|x| x * 2).collect();
    println!("Parallel map result: {:?}", data);
    
    // Test 3: Parallel chunks
    let slice = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let chunks: Vec<_> = par_chunks(&slice, 3).map(|chunk| chunk.len()).collect();
    println!("Chunk sizes: {:?}", chunks);
    
    // Test 4: Parallel bridge
    let result: Vec<i32> = slice.iter().par_bridge().map(|&x| x * x).collect();
    println!("Par bridge result: {:?}", result);
    
    // Test 5: parallel_map function
    let squared = parallel_map(&slice, |&x| x * x);
    println!("Parallel map function result: {:?}", squared);
    
    println!("All parallel operations imports are working correctly!");
}