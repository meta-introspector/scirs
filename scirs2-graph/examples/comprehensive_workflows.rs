//! Comprehensive workflow examples for scirs2-graph
//!
//! This example demonstrates common workflows and usage patterns
//! for the scirs2-graph library, showcasing various graph operations,
//! algorithms, and performance optimizations.

use rand::rngs::StdRng;
use rand::{thread_rng, SeedableRng};
use scirs2_graph::{
    barabasi_albert_graph,
    betweenness_centrality,
    breadth_first_search,
    // Graph measures
    centrality,
    clustering_coefficient,
    connected_components,
    // Algorithms
    dijkstra_path,
    // Graph generators
    erdos_renyi_graph,
    louvain_communities_result,
    pagerank_centrality,
    parallel_pagerank_centrality,
    watts_strogatz_graph,
    CentralityType,
    DiGraph,
    // Core graph types
    Graph,
    // I/O and utilities
    GraphError,
    Result,
};
use std::collections::HashMap;

/// Workflow 1: Basic Graph Operations and Analysis
///
/// Demonstrates fundamental graph creation, modification, and basic analysis.
fn workflow_basic_operations() -> Result<()> {
    println!("🔹 Workflow 1: Basic Graph Operations");

    // Create a new undirected graph
    let mut graph = Graph::<String, f64>::new();

    // Add nodes representing cities
    let cities = vec!["New York", "London", "Tokyo", "Sydney", "Paris"];
    for city in &cities {
        graph.add_node(city.to_string());
    }

    // Add weighted edges representing distances (in thousands of km)
    graph.add_edge("New York".to_string(), "London".to_string(), 5.5)?;
    graph.add_edge("New York".to_string(), "Tokyo".to_string(), 10.8)?;
    graph.add_edge("London".to_string(), "Paris".to_string(), 0.3)?;
    graph.add_edge("London".to_string(), "Sydney".to_string(), 17.0)?;
    graph.add_edge("Tokyo".to_string(), "Sydney".to_string(), 7.8)?;
    graph.add_edge("Paris".to_string(), "Sydney".to_string(), 16.8)?;

    // Basic graph properties
    println!("  📊 Graph Statistics:");
    println!("    Nodes: {}", graph.node_count());
    println!("    Edges: {}", graph.edge_count());
    println!("    Density: {:.3}", graph.density()?);

    // Find shortest path between cities
    if let Some(path) = dijkstra_path(&graph, &"New York".to_string(), &"Sydney".to_string())? {
        println!(
            "  🛣️  Shortest path NYC→Sydney: {:?} (distance: {:.1}k km)",
            path.nodes, path.total_weight
        );
    }

    // Calculate centrality measures
    let degree_centrality = centrality(&graph, CentralityType::Degree)?;
    let most_central = degree_centrality
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!(
        "  🎯 Most central city (degree): {} ({:.3})",
        most_central.0, most_central.1
    );

    println!("  ✅ Basic operations completed successfully\n");
    Ok(())
}

/// Workflow 2: Large Graph Analysis with Performance Optimizations
///
/// Demonstrates working with larger graphs and leveraging performance optimizations
/// like parallel processing for computationally intensive algorithms.
fn workflow_large_graph_analysis() -> Result<()> {
    println!("🔹 Workflow 2: Large Graph Analysis");

    // Generate a large scale-free network (social network-like)
    let num_nodes = 10_000;
    let mut rng = StdRng::seed_from_u64(42);

    println!(
        "  🏗️  Generating Barabási-Albert graph ({} nodes)...",
        num_nodes
    );
    let graph = barabasi_albert_graph(num_nodes, 3, &mut rng)?;

    println!("  📊 Large Graph Statistics:");
    println!("    Nodes: {}", graph.node_count());
    println!("    Edges: {}", graph.edge_count());
    println!(
        "    Average degree: {:.2}",
        2.0 * graph.edge_count() as f64 / graph.node_count() as f64
    );

    // Compare sequential vs parallel PageRank
    println!("  ⚡ PageRank Performance Comparison:");

    // Sequential PageRank
    let start = std::time::Instant::now();
    let sequential_pagerank = pagerank_centrality(&graph, 0.85, 1e-6)?;
    let sequential_time = start.elapsed();

    // Parallel PageRank (automatically uses parallel version for large graphs)
    let start = std::time::Instant::now();
    let parallel_pagerank = parallel_pagerank_centrality(&graph, 0.85, 1e-6, None)?;
    let parallel_time = start.elapsed();

    println!("    Sequential: {:.2}ms", sequential_time.as_millis());
    println!("    Parallel:   {:.2}ms", parallel_time.as_millis());

    if parallel_time < sequential_time {
        let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
        println!("    🚀 Speedup: {:.2}x", speedup);
    }

    // Find most influential nodes
    let top_nodes: Vec<_> = parallel_pagerank.iter().collect::<Vec<_>>();
    let mut sorted_nodes = top_nodes.clone();
    sorted_nodes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("  🌟 Top 5 most influential nodes (PageRank):");
    for (i, (node, score)) in sorted_nodes.iter().take(5).enumerate() {
        println!("    {}. Node {}: {:.6}", i + 1, node, score);
    }

    println!("  ✅ Large graph analysis completed\n");
    Ok(())
}

/// Workflow 3: Community Detection and Network Analysis
///
/// Demonstrates community detection algorithms and network structure analysis.
fn workflow_community_detection() -> Result<()> {
    println!("🔹 Workflow 3: Community Detection");

    // Generate a graph with known community structure
    let mut rng = StdRng::seed_from_u64(42);
    let graph = watts_strogatz_graph(1000, 6, 0.1, &mut rng)?;

    println!("  🏗️  Generated small-world network (Watts-Strogatz)");
    println!(
        "    Nodes: {}, Edges: {}",
        graph.node_count(),
        graph.edge_count()
    );

    // Detect communities using Louvain algorithm
    println!("  🔍 Detecting communities with Louvain algorithm...");
    let communities = louvain_communities_result(&graph);

    println!("  📊 Community Structure:");
    println!("    Number of communities: {}", communities.num_communities);

    if let Some(modularity) = communities.quality_score {
        println!("    Modularity: {:.4}", modularity);
    }

    // Analyze community sizes
    let mut sizes: Vec<usize> = communities.communities.iter().map(|c| c.len()).collect();
    sizes.sort_by(|a, b| b.cmp(a));

    println!("  📈 Community size distribution:");
    println!("    Largest community: {} nodes", sizes[0]);
    println!("    Smallest community: {} nodes", sizes[sizes.len() - 1]);
    println!(
        "    Average size: {:.1}",
        sizes.iter().sum::<usize>() as f64 / sizes.len() as f64
    );

    // Calculate clustering coefficient
    let clustering = clustering_coefficient(&graph)?;
    println!("  🔗 Network clustering coefficient: {:.4}", clustering);

    println!("  ✅ Community detection completed\n");
    Ok(())
}

/// Workflow 4: Multi-Graph Comparison and Benchmarking
///
/// Demonstrates comparing different graph types and their properties.
fn workflow_graph_comparison() -> Result<()> {
    println!("🔹 Workflow 4: Graph Type Comparison");

    let n = 1000;
    let mut rng = StdRng::seed_from_u64(42);

    // Generate different types of graphs
    println!("  🏗️  Generating different graph types ({} nodes each):", n);

    let random_graph = erdos_renyi_graph(n, 0.01, &mut rng)?;
    let scale_free_graph = barabasi_albert_graph(n, 3, &mut rng)?;
    let small_world_graph = watts_strogatz_graph(n, 6, 0.1, &mut rng)?;

    let graphs = vec![
        ("Random (Erdős-Rényi)", &random_graph),
        ("Scale-free (Barabási-Albert)", &scale_free_graph),
        ("Small-world (Watts-Strogatz)", &small_world_graph),
    ];

    println!("  📊 Comparative Analysis:");
    println!(
        "    {':<25'} {':<8'} {':<8'} {':<10'} {':<10'}",
        "Graph Type", "Edges", "Density", "Clustering", "Diameter"
    );
    println!("    {}", "-".repeat(70));

    for (name, graph) in graphs {
        let edges = graph.edge_count();
        let density = graph.density()?;
        let clustering = clustering_coefficient(graph)?;

        // Estimate diameter (computationally expensive for large graphs)
        let sample_nodes: Vec<_> = graph.nodes().into_iter().take(10).collect();
        let mut max_distance = 0.0;

        for i in 0..sample_nodes.len().min(5) {
            for j in (i + 1)..sample_nodes.len().min(5) {
                if let Some(path) = dijkstra_path(graph, &sample_nodes[i], &sample_nodes[j])? {
                    max_distance = max_distance.max(path.nodes.len() as f64 - 1.0);
                }
            }
        }

        println!(
            "    {:25} {:8} {:8.4} {:10.4} {:10.1}",
            name, edges, density, clustering, max_distance
        );
    }

    println!("  ✅ Graph comparison completed\n");
    Ok(())
}

/// Workflow 5: Directed Graph Analysis
///
/// Demonstrates working with directed graphs and specific directed graph algorithms.
fn workflow_directed_graph_analysis() -> Result<()> {
    println!("🔹 Workflow 5: Directed Graph Analysis");

    // Create a directed graph representing a citation network
    let mut digraph = DiGraph::<String, f64>::new();

    let papers = vec![
        "Paper A", "Paper B", "Paper C", "Paper D", "Paper E", "Paper F", "Paper G", "Paper H",
        "Paper I", "Paper J",
    ];

    // Add papers as nodes
    for paper in &papers {
        digraph.add_node(paper.to_string());
    }

    // Add citation relationships (directed edges)
    let citations = vec![
        ("Paper A", "Paper B"),
        ("Paper A", "Paper C"),
        ("Paper B", "Paper D"),
        ("Paper C", "Paper D"),
        ("Paper D", "Paper E"),
        ("Paper E", "Paper F"),
        ("Paper F", "Paper G"),
        ("Paper G", "Paper H"),
        ("Paper H", "Paper I"),
        ("Paper I", "Paper J"),
        ("Paper J", "Paper A"), // Creates a cycle
    ];

    for (citing, cited) in citations {
        digraph.add_edge(citing.to_string(), cited.to_string(), 1.0)?;
    }

    println!("  📚 Citation Network Statistics:");
    println!("    Papers: {}", digraph.node_count());
    println!("    Citations: {}", digraph.edge_count());

    // Analyze in-degree and out-degree distributions
    let nodes = digraph.nodes();
    let mut in_degrees = Vec::new();
    let mut out_degrees = Vec::new();

    for node in &nodes {
        in_degrees.push(digraph.in_degree(node));
        out_degrees.push(digraph.out_degree(node));
    }

    let avg_in_degree = in_degrees.iter().sum::<usize>() as f64 / nodes.len() as f64;
    let avg_out_degree = out_degrees.iter().sum::<usize>() as f64 / nodes.len() as f64;

    println!("    Average in-degree: {:.2}", avg_in_degree);
    println!("    Average out-degree: {:.2}", avg_out_degree);

    // Find strongly connected components
    let sccs = strongly_connected_components(&digraph);
    println!("    Strongly connected components: {}", sccs.len());

    println!("  ✅ Directed graph analysis completed\n");
    Ok(())
}

/// Main function demonstrating all workflows
fn main() -> Result<()> {
    println!("🚀 SciRS2-Graph Comprehensive Workflow Examples");
    println!("===============================================\n");

    // Run all workflow examples
    workflow_basic_operations()?;
    workflow_large_graph_analysis()?;
    workflow_community_detection()?;
    workflow_graph_comparison()?;
    workflow_directed_graph_analysis()?;

    println!("🎉 All workflows completed successfully!");
    println!("\n💡 Tips for optimal performance:");
    println!("   • Use parallel algorithms for graphs with >10,000 nodes");
    println!("   • Enable SIMD optimizations with appropriate CPU features");
    println!("   • Consider memory-mapped storage for very large graphs");
    println!("   • Profile your specific use case to identify bottlenecks");

    Ok(())
}
