# Extended Usage Examples for scirs2-graph

This guide provides comprehensive examples for common graph processing workflows using scirs2-graph.

## Table of Contents

1. [Basic Graph Operations](#basic-graph-operations)
2. [Social Network Analysis](#social-network-analysis)
3. [Route Finding and Navigation](#route-finding-and-navigation)
4. [Community Detection](#community-detection)
5. [Graph Machine Learning](#graph-machine-learning)
6. [Bioinformatics Applications](#bioinformatics-applications)
7. [Network Flow Problems](#network-flow-problems)
8. [Graph Visualization Preparation](#graph-visualization-preparation)

## Basic Graph Operations

### Creating and Manipulating Graphs

```rust
use scirs2_graph::{Graph, DiGraph, Node, EdgeWeight};

// Create an undirected graph
let mut graph = Graph::<String, f64>::new();

// Add nodes
let alice = graph.add_node("Alice".to_string());
let bob = graph.add_node("Bob".to_string());
let charlie = graph.add_node("Charlie".to_string());

// Add weighted edges
graph.add_edge("Alice".to_string(), "Bob".to_string(), 1.5)?;
graph.add_edge("Bob".to_string(), "Charlie".to_string(), 2.0)?;
graph.add_edge("Alice".to_string(), "Charlie".to_string(), 3.5)?;

// Query the graph
println!("Number of nodes: {}", graph.node_count());
println!("Number of edges: {}", graph.edge_count());
println!("Degree of Bob: {}", graph.degree(&"Bob".to_string()).unwrap());

// Check if edge exists
if graph.has_edge(&"Alice".to_string(), &"Bob".to_string()).unwrap() {
    println!("Alice and Bob are connected");
}

// Get neighbors
let neighbors = graph.neighbors(&"Bob".to_string())?;
println!("Bob's neighbors: {:?}", neighbors);
```

### Working with Directed Graphs

```rust
use scirs2_graph::{DiGraph, shortest_path};

// Create a directed graph for a web link structure
let mut web = DiGraph::<&str, f64>::new();

// Add pages
let pages = vec!["home", "about", "blog", "contact", "post1", "post2"];
for page in &pages {
    web.add_node(*page);
}

// Add links (directed edges)
web.add_edge("home", "about", 1.0)?;
web.add_edge("home", "blog", 1.0)?;
web.add_edge("home", "contact", 1.0)?;
web.add_edge("blog", "post1", 1.0)?;
web.add_edge("blog", "post2", 1.0)?;
web.add_edge("post1", "post2", 1.0)?;
web.add_edge("about", "contact", 1.0)?;

// Find paths
match shortest_path(&web, &"home", &"post2") {
    Ok((path, distance)) => {
        println!("Path from home to post2: {:?}", path);
        println!("Number of clicks: {}", distance);
    }
    Err(e) => println!("No path found: {}", e),
}

// Analyze in-degree and out-degree
println!("Page statistics:");
for page in &pages {
    let in_deg = web.in_degree(page).unwrap_or(0);
    let out_deg = web.out_degree(page).unwrap_or(0);
    println!("{}: {} incoming links, {} outgoing links", page, in_deg, out_deg);
}
```

## Social Network Analysis

### Finding Influential Users

```rust
use scirs2_graph::{
    Graph, barabasi_albert_graph, 
    pagerank_centrality, betweenness_centrality, 
    eigenvector_centrality, closeness_centrality
};
use rand::SeedableRng;
use rand::rngs::StdRng;

// Generate a social network (scale-free network)
let mut rng = StdRng::seed_from_u64(42);
let social_network = barabasi_albert_graph(100, 3, &mut rng)?;

// Calculate various centrality measures
let pagerank = pagerank_centrality(&social_network, 0.85, 1e-6)?;
let betweenness = betweenness_centrality(&social_network, true)?;
let eigenvector = eigenvector_centrality(&social_network)?;
let closeness = closeness_centrality(&social_network)?;

// Find top 5 influential users by different metrics
println!("Top 5 users by PageRank:");
let mut pr_vec: Vec<_> = pagerank.iter().collect();
pr_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
for (user, score) in pr_vec.iter().take(5) {
    println!("  User {}: {:.4}", user, score);
}

println!("\nTop 5 users by Betweenness Centrality:");
let mut bt_vec: Vec<_> = betweenness.iter().collect();
bt_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
for (user, score) in bt_vec.iter().take(5) {
    println!("  User {}: {:.4}", user, score);
}
```

### Community Detection

```rust
use scirs2_graph::{Graph, louvain_communities, modularity};

// Assuming we have a social network graph
let communities = louvain_communities(&social_network)?;

// Print community assignments
println!("Found {} communities", communities.len());
for (idx, community) in communities.iter().enumerate() {
    println!("Community {}: {} members", idx, community.len());
}

// Calculate modularity to assess community quality
let modularity_score = modularity(&social_network, &communities)?;
println!("Modularity score: {:.4}", modularity_score);

// Find bridges between communities
let mut inter_community_edges = 0;
let mut intra_community_edges = 0;

// Create a map from node to community
let mut node_to_community = std::collections::HashMap::new();
for (comm_idx, community) in communities.iter().enumerate() {
    for node in community {
        node_to_community.insert(node, comm_idx);
    }
}

// Count edges
for edge in social_network.edges() {
    let (src, tgt, _) = edge;
    if node_to_community.get(&src) == node_to_community.get(&tgt) {
        intra_community_edges += 1;
    } else {
        inter_community_edges += 1;
    }
}

println!("Intra-community edges: {}", intra_community_edges);
println!("Inter-community edges: {}", inter_community_edges);
```

## Route Finding and Navigation

### Finding Optimal Routes in a Transportation Network

```rust
use scirs2_graph::{DiGraph, shortest_path, k_shortest_paths, astar_search};
use std::collections::HashMap;

// Create a transportation network
let mut transport = DiGraph::<&str, f64>::new();

// Add cities
let cities = vec![
    "New York", "Boston", "Philadelphia", "Washington",
    "Chicago", "Detroit", "Cleveland", "Pittsburgh"
];

for city in &cities {
    transport.add_node(*city);
}

// Add routes with distances (in miles)
let routes = vec![
    ("New York", "Boston", 215.0),
    ("New York", "Philadelphia", 95.0),
    ("Philadelphia", "Washington", 139.0),
    ("New York", "Pittsburgh", 371.0),
    ("Pittsburgh", "Cleveland", 134.0),
    ("Cleveland", "Detroit", 170.0),
    ("Cleveland", "Chicago", 345.0),
    ("Pittsburgh", "Washington", 241.0),
];

for (src, dst, distance) in routes {
    transport.add_edge(src, dst, distance)?;
    transport.add_edge(dst, src, distance)?; // Make bidirectional
}

// Find shortest route
let (path, distance) = shortest_path(&transport, &"New York", &"Chicago")?;
println!("Shortest route from New York to Chicago:");
println!("  Path: {:?}", path.join(" -> "));
println!("  Distance: {} miles", distance);

// Find alternative routes
let alternatives = k_shortest_paths(&transport, &"New York", &"Chicago", 3)?;
println!("\nTop 3 routes from New York to Chicago:");
for (idx, (path, dist)) in alternatives.iter().enumerate() {
    println!("  Route {}: {} miles via {:?}", idx + 1, dist, path);
}

// Use A* with geographical heuristic
// Approximate coordinates for heuristic
let coords: HashMap<&str, (f64, f64)> = vec![
    ("New York", (40.7128, -74.0060)),
    ("Boston", (42.3601, -71.0589)),
    ("Philadelphia", (39.9526, -75.1652)),
    ("Washington", (38.9072, -77.0369)),
    ("Chicago", (41.8781, -87.6298)),
    ("Detroit", (42.3314, -83.0458)),
    ("Cleveland", (41.4993, -81.6944)),
    ("Pittsburgh", (40.4406, -79.9959)),
].into_iter().collect();

// Haversine distance heuristic
let heuristic = |city: &&str| -> f64 {
    let (lat1, lon1) = coords[city];
    let (lat2, lon2) = coords["Chicago"]; // Target
    
    let r = 3959.0; // Earth radius in miles
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat/2.0).sin().powi(2) + 
            lat1.to_radians().cos() * lat2.to_radians().cos() * 
            (dlon/2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    r * c
};

// Find route using A*
let (astar_path, astar_dist) = astar_search(
    &transport, 
    &"New York", 
    &"Chicago", 
    heuristic
)?;
println!("\nA* route from New York to Chicago:");
println!("  Path: {:?}", astar_path.join(" -> "));
println!("  Distance: {} miles", astar_dist);
```

## Community Detection

### Analyzing Research Collaboration Networks

```rust
use scirs2_graph::{
    Graph, AttributedGraph, 
    louvain_communities, label_propagation,
    infomap_communities, hierarchical_communities
};

// Create a collaboration network with attributes
let mut collab = AttributedGraph::<String, f64>::new();

// Add researchers with attributes
let researchers = vec![
    ("Dr. Smith", "Machine Learning"),
    ("Dr. Jones", "Machine Learning"),
    ("Dr. Brown", "Computer Vision"),
    ("Dr. Davis", "Computer Vision"),
    ("Dr. Wilson", "NLP"),
    ("Dr. Taylor", "NLP"),
    ("Dr. Anderson", "Machine Learning"),
    ("Dr. Thomas", "Computer Vision"),
];

for (name, field) in researchers {
    let mut attrs = std::collections::HashMap::new();
    attrs.insert("field".to_string(), field.to_string().into());
    attrs.insert("papers".to_string(), (rand::random::<f64>() * 50.0).into());
    collab.add_node_with_attrs(name.to_string(), attrs);
}

// Add collaborations (edges represent joint papers)
let collaborations = vec![
    ("Dr. Smith", "Dr. Jones", 5),
    ("Dr. Smith", "Dr. Anderson", 3),
    ("Dr. Jones", "Dr. Anderson", 4),
    ("Dr. Brown", "Dr. Davis", 6),
    ("Dr. Brown", "Dr. Thomas", 2),
    ("Dr. Davis", "Dr. Thomas", 4),
    ("Dr. Wilson", "Dr. Taylor", 7),
    ("Dr. Smith", "Dr. Brown", 1),  // Cross-field collaboration
    ("Dr. Davis", "Dr. Wilson", 1), // Cross-field collaboration
];

for (a, b, papers) in collaborations {
    let mut attrs = std::collections::HashMap::new();
    attrs.insert("joint_papers".to_string(), (papers as f64).into());
    collab.add_edge_with_attrs(
        a.to_string(), 
        b.to_string(), 
        papers as f64,
        attrs
    )?;
}

// Compare different community detection algorithms
println!("Community Detection Results:\n");

// 1. Louvain Method
let louvain_comms = louvain_communities(&collab)?;
println!("Louvain Method found {} communities", louvain_comms.len());

// 2. Label Propagation
let label_prop_comms = label_propagation(&collab)?;
println!("Label Propagation found {} communities", label_prop_comms.len());

// 3. Infomap
let infomap_comms = infomap_communities(&collab)?;
println!("Infomap found {} communities", infomap_comms.len());

// Analyze community composition
println!("\nLouvain Community Composition:");
for (idx, community) in louvain_comms.iter().enumerate() {
    println!("Community {}:", idx);
    let mut field_count = std::collections::HashMap::new();
    
    for member in community {
        if let Some(field) = collab.node_attr(member, "field")? {
            if let AttributeValue::String(f) = field {
                *field_count.entry(f).or_insert(0) += 1;
            }
        }
    }
    
    for (field, count) in field_count {
        println!("  {}: {} researchers", field, count);
    }
}
```

## Graph Machine Learning

### Node Classification with Graph Features

```rust
use scirs2_graph::{Graph, centrality, CentralityType, clustering_coefficient};
use ndarray::{Array2, s};

// Create a graph and extract features for machine learning
fn extract_node_features(graph: &Graph<usize, f64>) -> Array2<f64> {
    let n = graph.node_count();
    let mut features = Array2::zeros((n, 7)); // 7 features per node
    
    // Feature 1: Degree
    let degrees = graph.degree_vector();
    features.slice_mut(s![.., 0]).assign(&degrees.mapv(|d| d as f64));
    
    // Feature 2: Clustering coefficient
    let clustering = clustering_coefficient(graph).unwrap();
    for (i, &node) in graph.nodes().iter().enumerate() {
        features[[i, 1]] = clustering.get(&node).copied().unwrap_or(0.0);
    }
    
    // Feature 3-6: Various centrality measures
    let pagerank = pagerank_centrality(graph, 0.85, 1e-6).unwrap();
    let betweenness = betweenness_centrality(graph, true).unwrap();
    let closeness = closeness_centrality(graph).unwrap();
    let eigenvector = eigenvector_centrality(graph).unwrap();
    
    for (i, &node) in graph.nodes().iter().enumerate() {
        features[[i, 2]] = pagerank.get(&node).copied().unwrap_or(0.0);
        features[[i, 3]] = betweenness.get(&node).copied().unwrap_or(0.0);
        features[[i, 4]] = closeness.get(&node).copied().unwrap_or(0.0);
        features[[i, 5]] = eigenvector.get(&node).copied().unwrap_or(0.0);
    }
    
    // Feature 7: Average neighbor degree
    for (i, &node) in graph.nodes().iter().enumerate() {
        let neighbors = graph.neighbors(&node).unwrap();
        if !neighbors.is_empty() {
            let avg_neighbor_degree: f64 = neighbors.iter()
                .map(|n| graph.degree(n).unwrap_or(0) as f64)
                .sum::<f64>() / neighbors.len() as f64;
            features[[i, 6]] = avg_neighbor_degree;
        }
    }
    
    features
}

// Example: Create a graph and extract features
let mut rng = StdRng::seed_from_u64(42);
let graph = erdos_renyi_graph(50, 0.1, &mut rng)?;
let node_features = extract_node_features(&graph);

println!("Extracted features shape: {:?}", node_features.shape());
println!("First 5 nodes features:");
for i in 0..5 {
    println!("Node {}: {:?}", i, node_features.row(i));
}
```

### Graph Embeddings with Random Walk

```rust
use scirs2_graph::{Graph, RandomWalkGenerator, Node2Vec, Node2VecConfig};

// Configure Node2Vec for graph embeddings
let config = Node2VecConfig {
    dimensions: 64,
    walk_length: 80,
    num_walks: 10,
    p: 1.0,  // Return parameter
    q: 0.5,  // In-out parameter (q < 1 encourages exploration)
    window_size: 10,
    min_count: 1,
    epochs: 5,
};

// Generate embeddings
let embeddings = Node2Vec::new(config).fit(&graph)?;

// Use embeddings for downstream tasks
println!("Generated {} embeddings of dimension {}", 
         embeddings.len(), 
         embeddings.values().next().unwrap().len());

// Find similar nodes based on embeddings
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b)
}

// Find most similar nodes to node 0
let node0_embedding = &embeddings[&0];
let mut similarities: Vec<(usize, f64)> = embeddings.iter()
    .filter(|(&n, _)| n != 0)
    .map(|(&n, emb)| (n, cosine_similarity(node0_embedding, emb)))
    .collect();

similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

println!("\nMost similar nodes to node 0:");
for (node, sim) in similarities.iter().take(5) {
    println!("  Node {}: similarity = {:.4}", node, sim);
}
```

## Bioinformatics Applications

### Protein Interaction Networks

```rust
use scirs2_graph::{Graph, AttributedGraph, connected_components, k_core_decomposition};

// Create a protein-protein interaction network
let mut ppi = AttributedGraph::<String, f64>::new();

// Add proteins with functional annotations
let proteins = vec![
    ("P53", "tumor suppressor"),
    ("MDM2", "E3 ubiquitin ligase"),
    ("BRCA1", "DNA repair"),
    ("BRCA2", "DNA repair"),
    ("ATM", "kinase"),
    ("CHK2", "kinase"),
    ("RAD51", "DNA repair"),
];

for (protein, function) in proteins {
    let mut attrs = std::collections::HashMap::new();
    attrs.insert("function".to_string(), function.to_string().into());
    attrs.insert("essential".to_string(), (rand::random::<bool>()).into());
    ppi.add_node_with_attrs(protein.to_string(), attrs);
}

// Add interactions
let interactions = vec![
    ("P53", "MDM2", 0.95),    // Strong interaction
    ("P53", "ATM", 0.8),
    ("P53", "CHK2", 0.75),
    ("BRCA1", "BRCA2", 0.9),
    ("BRCA1", "RAD51", 0.85),
    ("BRCA2", "RAD51", 0.88),
    ("ATM", "CHK2", 0.82),
    ("ATM", "P53", 0.8),
];

for (a, b, confidence) in interactions {
    ppi.add_edge(a.to_string(), b.to_string(), confidence)?;
}

// Find protein complexes (dense subgraphs)
let k_cores = k_core_decomposition(&ppi)?;
println!("K-core decomposition:");
for (k, nodes) in k_cores {
    if nodes.len() > 1 {
        println!("  {}-core: {:?}", k, nodes);
    }
}

// Identify essential proteins using centrality
let betweenness = betweenness_centrality(&ppi, true)?;
let degree_cent = centrality(&ppi, CentralityType::Degree)?;

println!("\nProtein importance analysis:");
for protein in ppi.nodes() {
    let deg = degree_cent.get(protein).unwrap_or(&0.0);
    let bet = betweenness.get(protein).unwrap_or(&0.0);
    println!("  {}: degree={:.2}, betweenness={:.4}", protein, deg, bet);
}

// Find functional modules
let modules = louvain_communities(&ppi)?;
println!("\nFunctional modules:");
for (idx, module) in modules.iter().enumerate() {
    print!("  Module {}: ", idx);
    let functions: Vec<String> = module.iter()
        .filter_map(|p| {
            ppi.node_attr(p, "function").ok()
                .and_then(|attr| match attr {
                    AttributeValue::String(s) => Some(s),
                    _ => None,
                })
        })
        .collect();
    println!("{:?}", functions);
}
```

## Network Flow Problems

### Supply Chain Optimization

```rust
use scirs2_graph::{DiGraph, dinic_max_flow, minimum_cut};

// Create a supply chain network
let mut supply_chain = DiGraph::<&str, f64>::new();

// Add nodes: sources (factories), intermediate (warehouses), sinks (stores)
let nodes = vec![
    "Factory1", "Factory2",           // Sources
    "Warehouse1", "Warehouse2", "Warehouse3",  // Intermediate
    "Store1", "Store2", "Store3", "Store4",    // Sinks
    "Source", "Sink"                  // Super source/sink
];

for node in &nodes {
    supply_chain.add_node(*node);
}

// Connect super source to factories
supply_chain.add_edge("Source", "Factory1", 100.0)?;  // Capacity
supply_chain.add_edge("Source", "Factory2", 150.0)?;

// Factory to warehouse connections
supply_chain.add_edge("Factory1", "Warehouse1", 60.0)?;
supply_chain.add_edge("Factory1", "Warehouse2", 40.0)?;
supply_chain.add_edge("Factory2", "Warehouse2", 70.0)?;
supply_chain.add_edge("Factory2", "Warehouse3", 80.0)?;

// Warehouse to store connections
supply_chain.add_edge("Warehouse1", "Store1", 30.0)?;
supply_chain.add_edge("Warehouse1", "Store2", 30.0)?;
supply_chain.add_edge("Warehouse2", "Store2", 40.0)?;
supply_chain.add_edge("Warehouse2", "Store3", 30.0)?;
supply_chain.add_edge("Warehouse3", "Store3", 40.0)?;
supply_chain.add_edge("Warehouse3", "Store4", 40.0)?;

// Stores to super sink (demand)
supply_chain.add_edge("Store1", "Sink", 30.0)?;
supply_chain.add_edge("Store2", "Sink", 60.0)?;
supply_chain.add_edge("Store3", "Sink", 55.0)?;
supply_chain.add_edge("Store4", "Sink", 35.0)?;

// Calculate maximum flow
let (max_flow, flow_values) = dinic_max_flow(&supply_chain, &"Source", &"Sink")?;
println!("Maximum flow through supply chain: {}", max_flow);

// Analyze bottlenecks
println!("\nFlow utilization:");
for ((src, dst), flow) in &flow_values {
    if let Some(edge) = supply_chain.find_edge(src, dst) {
        let capacity = supply_chain.edge_weight(edge).unwrap();
        let utilization = flow / capacity * 100.0;
        if utilization > 80.0 {  // Potential bottleneck
            println!("  {} -> {}: {:.0}/{:.0} ({:.1}%) ⚠️ HIGH UTILIZATION", 
                     src, dst, flow, capacity, utilization);
        }
    }
}

// Find minimum cut (critical edges)
let (cut_value, cut_edges) = minimum_cut(&supply_chain, &"Source", &"Sink")?;
println!("\nMinimum cut value: {}", cut_value);
println!("Critical edges (removing these disconnects supply):");
for (src, dst) in cut_edges {
    println!("  {} -> {}", src, dst);
}
```

## Graph Visualization Preparation

### Preparing Graph Data for Visualization

```rust
use scirs2_graph::{
    Graph, spring_layout, circular_layout, 
    hierarchical_layout, spectral_layout,
    write_graphml, write_gml
};

// Create a sample graph
let mut viz_graph = Graph::<String, f64>::new();

// Add nodes with labels
let nodes = vec!["A", "B", "C", "D", "E", "F", "G", "H"];
for node in &nodes {
    viz_graph.add_node(node.to_string());
}

// Add edges to create interesting structure
let edges = vec![
    ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"),
    ("D", "E"), ("E", "F"), ("E", "G"), ("F", "G"),
    ("G", "H"), ("F", "H"),
];

for (src, dst) in edges {
    viz_graph.add_edge(src.to_string(), dst.to_string(), 1.0)?;
}

// Generate different layouts
let layouts = vec![
    ("spring", spring_layout(&viz_graph, 1000, 1.0)?),
    ("circular", circular_layout(&viz_graph)?),
    ("spectral", spectral_layout(&viz_graph)?),
    ("hierarchical", hierarchical_layout(&viz_graph, &"A".to_string())?),
];

// Print layout coordinates
for (layout_name, positions) in &layouts {
    println!("\n{} layout:", layout_name);
    for (node, pos) in positions {
        println!("  {}: ({:.3}, {:.3})", node, pos.x, pos.y);
    }
}

// Export for visualization tools
write_graphml(&viz_graph, "graph_for_viz.graphml")?;
write_gml(&viz_graph, "graph_for_viz.gml")?;

// Create a more complex attributed graph for rich visualization
let mut rich_graph = AttributedGraph::<String, f64>::new();

// Add nodes with multiple attributes
for (i, node) in nodes.iter().enumerate() {
    let mut attrs = std::collections::HashMap::new();
    attrs.insert("size".to_string(), ((i + 1) as f64 * 10.0).into());
    attrs.insert("color".to_string(), format!("#{:06x}", i * 0x223344).into());
    attrs.insert("shape".to_string(), 
        if i % 2 == 0 { "circle" } else { "square" }.to_string().into());
    rich_graph.add_node_with_attrs(node.to_string(), attrs);
}

// Add edges with weights as thickness
for (src, dst) in edges {
    let weight = rand::random::<f64>() * 5.0 + 1.0;
    let mut attrs = std::collections::HashMap::new();
    attrs.insert("thickness".to_string(), weight.into());
    attrs.insert("style".to_string(), 
        if weight > 3.0 { "solid" } else { "dashed" }.to_string().into());
    rich_graph.add_edge_with_attrs(
        src.to_string(), 
        dst.to_string(), 
        weight,
        attrs
    )?;
}

// Export rich graph
write_graphml(&rich_graph, "rich_graph_for_viz.graphml")?;
println!("\nExported graphs for visualization!");
```

## Performance Tips

### Working with Large Graphs

```rust
use scirs2_graph::{Graph, LargeGraphOps, StreamingGraphProcessor, ParallelConfig};

// Configure parallel processing
let config = ParallelConfig {
    num_threads: num_cpus::get(),
    chunk_size: 1000,
    enable_work_stealing: true,
};

// Process large graph in streaming fashion
let processor = StreamingGraphProcessor::new(config);

// Example: Process edges in batches
processor.process_edges_batched(&large_graph, 10000, |batch| {
    // Process batch of edges
    for (src, dst, weight) in batch {
        // Perform computation
    }
})?;

// Use memory-efficient algorithms for large graphs
let components = large_graph.connected_components_streaming()?;
let pagerank = large_graph.pagerank_approximate(0.85, 0.01)?;

// Enable performance monitoring
let monitor = PerformanceMonitor::new();
monitor.start("algorithm_execution");

// Run algorithm
let result = some_expensive_algorithm(&large_graph)?;

let stats = monitor.stop("algorithm_execution");
println!("Execution time: {:?}", stats.duration);
println!("Memory usage: {} MB", stats.memory_used / 1_000_000);
```

## Error Handling Best Practices

```rust
use scirs2_graph::{Graph, GraphError, Result};

fn safe_graph_operation() -> Result<()> {
    let mut graph = Graph::<String, f64>::new();
    
    // Always handle potential errors
    match graph.add_edge("A".to_string(), "B".to_string(), 1.0) {
        Ok(_) => println!("Edge added successfully"),
        Err(GraphError::NodeNotFound(node)) => {
            println!("Node {} not found, adding it first", node);
            graph.add_node(node);
        }
        Err(e) => return Err(e),
    }
    
    // Use ? operator for propagation
    let path = shortest_path(&graph, &"A".to_string(), &"B".to_string())?;
    
    // Validate graph properties
    if graph.is_connected()? {
        println!("Graph is connected");
    } else {
        println!("Graph has {} components", connected_components(&graph)?.len());
    }
    
    Ok(())
}
```

## Summary

These examples demonstrate the versatility of scirs2-graph for various graph processing tasks:

- **Social Network Analysis**: Centrality measures, community detection
- **Route Finding**: Shortest paths, A* search, alternative routes
- **Machine Learning**: Feature extraction, graph embeddings
- **Bioinformatics**: Protein networks, functional modules
- **Operations Research**: Network flow, supply chain optimization
- **Visualization**: Layout algorithms, data export

The library provides efficient implementations suitable for both research and production use, with comprehensive error handling and performance optimization features.