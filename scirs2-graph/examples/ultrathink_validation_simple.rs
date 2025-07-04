//! Simple Ultrathink Mode Validation
//!
//! This example demonstrates and validates basic ultrathink mode functionality.

use rand::rng;
use scirs2_graph::{
    algorithms::{connected_components, dijkstra_path},
    base::Graph,
    generators::erdos_renyi_graph,
    measures::pagerank_centrality,
    ultrathink::{create_ultrathink_processor, execute_with_ultrathink},
    Result,
};
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🧪 Simple Ultrathink Mode Validation");
    println!("====================================");

    // Create a small test graph
    let mut rng = rand::rng();
    let graph = erdos_renyi_graph(100, 0.1, &mut rng)?;

    println!("✅ Generated test graph:");
    println!("   - Nodes: {}", graph.node_count());
    println!("   - Edges: {}", graph.edge_count());

    // Create advanced processor
    let mut processor = create_advanced_processor();
    println!("🚀 Ultrathink processor initialized");

    // Test 1: PageRank with advanced optimization
    println!("\n🧠 Test 1: PageRank Centrality");
    let start = Instant::now();
    let _pagerank_result =
        execute_with_advanced(&mut processor, &graph, "pagerank_centrality", |g| {
            pagerank_centrality(g, 0.85, 1e-6)
        })?;
    let duration = start.elapsed();
    println!("   ✅ Completed in {:?}", duration);

    // Test 2: Connected components with advanced optimization
    println!("\n🔗 Test 2: Connected Components");
    let start = Instant::now();
    let _components = execute_with_advanced(&mut processor, &graph, "connected_components", |g| {
        connected_components(g)
    })?;
    let duration = start.elapsed();
    println!("   ✅ Completed in {:?}", duration);

    // Test 3: Shortest path with advanced optimization
    println!("\n🛣️  Test 3: Shortest Path");
    let nodes: Vec<_> = graph.nodes().into_iter().collect();
    if nodes.len() >= 2 {
        let start = Instant::now();
        let _path_result = execute_with_advanced(&mut processor, &graph, "dijkstra_path", |g| {
            dijkstra_path(g, &nodes[0], &nodes[1])
        })?;
        let duration = start.elapsed();
        println!("   ✅ Completed in {:?}", duration);
    }

    // Get optimization statistics
    let stats = processor.get_optimization_stats();
    println!("\n📊 Optimization Statistics:");
    println!("   - Total optimizations: {}", stats.total_optimizations);
    println!("   - Average speedup: {:.2}x", stats.average_speedup);
    println!(
        "   - GPU utilization: {:.1}%",
        stats.gpu_utilization * 100.0
    );
    println!(
        "   - Memory efficiency: {:.1}%",
        stats.memory_efficiency * 100.0
    );

    println!("\n🎉 All advanced mode tests passed!");
    Ok(())
}
