# API Stability Review for scirs2-graph v1.0

This document provides a comprehensive review of the scirs2-graph API to ensure stability before the 1.0 release.

## Executive Summary

The scirs2-graph API is largely stable and ready for 1.0 release. This review identifies areas that require final adjustments and documents any breaking changes planned before 1.0.

## Review Methodology

1. API surface area analysis
2. Comparison with established graph libraries (NetworkX, igraph)  
3. User feedback incorporation
4. Performance implications of API choices
5. Future extensibility considerations

## Core API Components

### Graph Types

#### Current Status: Stable ✅
```rust
pub struct Graph<N, E> { ... }
pub struct DiGraph<N, E> { ... }
pub struct MultiGraph<N, E> { ... }
pub struct MultiDiGraph<N, E> { ... }
pub struct Hypergraph<N, E> { ... }
pub struct BipartiteGraph<N, E> { ... }
```

**Decision**: No changes needed. Generic parameters provide flexibility while maintaining type safety.

### Node and Edge Types

#### Current Status: Needs Review ⚠️
```rust
pub type Node = usize;
pub type EdgeWeight = f64;
```

**Issue**: Fixed types limit flexibility compared to NetworkX's any-hashable approach.

**Recommendation for 1.0**: 
- Keep current types for performance
- Document clearly in migration guide
- Consider generic node types in 2.0 if demand exists

### Error Handling

#### Current Status: Stable ✅
```rust
pub type Result<T> = std::result::Result<T, GraphError>;

pub enum GraphError {
    NodeNotFound(usize),
    EdgeNotFound(usize, usize),
    InvalidOperation(String),
    // ...
}
```

**Decision**: Error types are comprehensive and follow Rust best practices.

## Algorithm APIs

### Traversal Algorithms

#### Current Status: Stable ✅
```rust
pub fn breadth_first_search<N, E>(graph: &Graph<N, E>, start: &Node) -> Vec<Node>
pub fn depth_first_search<N, E>(graph: &Graph<N, E>, start: &Node) -> Vec<Node>
```

**Decision**: Signatures are clear and consistent.

### Shortest Path Algorithms

#### Current Status: Needs Minor Adjustment ⚠️
```rust
pub fn shortest_path<N, E>(graph: &Graph<N, E>, start: &Node, end: &Node) -> Option<(Vec<Node>, E)>
pub fn dijkstra<N, E>(graph: &Graph<N, E>, start: &Node) -> HashMap<Node, E>
```

**Issue**: Inconsistent naming (shortest_path vs dijkstra)

**Recommendation for 1.0**:
- Rename `shortest_path` to `dijkstra_path` for clarity
- Add deprecated alias for backward compatibility

### Centrality Measures

#### Current Status: Stable ✅
```rust
pub fn betweenness_centrality<N, E>(graph: &Graph<N, E>, normalized: bool) -> HashMap<Node, f64>
pub fn closeness_centrality<N, E>(graph: &Graph<N, E>) -> HashMap<Node, f64>
pub fn eigenvector_centrality<N, E>(graph: &Graph<N, E>) -> Result<HashMap<Node, f64>>
pub fn pagerank_centrality<N, E>(graph: &Graph<N, E>, damping: f64, tolerance: f64) -> Result<HashMap<Node, f64>>
```

**Decision**: APIs are consistent and well-designed.

### Community Detection

#### Current Status: Needs Standardization ⚠️
```rust
pub fn louvain_communities<N, E>(graph: &Graph<N, E>) -> Vec<HashSet<Node>>
pub fn label_propagation<N, E>(graph: &Graph<N, E>) -> HashMap<Node, usize>
pub fn infomap_communities<N, E>(graph: &Graph<N, E>) -> InfomapResult
```

**Issue**: Inconsistent return types across community detection algorithms

**Recommendation for 1.0**:
- Standardize on `CommunityResult` type that can represent different formats
- Provide conversion methods between formats

## Breaking Changes for 1.0

### 1. Algorithm Return Type Standardization

**Current**:
```rust
pub fn connected_components<N, E>(graph: &Graph<N, E>) -> Vec<HashSet<Node>>
```

**Proposed**:
```rust
pub fn connected_components<N, E>(graph: &Graph<N, E>) -> ComponentResult<Node>

pub struct ComponentResult<N> {
    components: Vec<HashSet<N>>,
    component_map: HashMap<N, usize>,
    num_components: usize,
}
```

**Migration Path**: Provide `.into_vec()` method for backward compatibility

### 2. Consistent Builder Pattern

**Current**: Mixed approaches for graph construction

**Proposed**:
```rust
let graph = Graph::builder()
    .directed(false)
    .with_capacity(1000, 5000)
    .build();
```

### 3. Async Algorithm Variants

**Current**: All algorithms are synchronous

**Proposed**: Add async variants for I/O-bound operations
```rust
pub async fn load_graph_async(path: &Path) -> Result<Graph<N, E>>
pub async fn save_graph_async<N, E>(graph: &Graph<N, E>, path: &Path) -> Result<()>
```

## Deprecation Schedule

### Phase 1 (v0.1.0-beta.2)
- Add deprecation warnings to affected APIs
- Introduce new APIs alongside old ones
- Update documentation with migration examples

### Phase 2 (v1.0.0-rc.1)
- Old APIs marked with `#[deprecated]` attribute
- Migration guide prominently featured
- Automated migration tool provided

### Phase 3 (v1.0.0)
- Old APIs remain but generate warnings
- Clear timeline for removal in 2.0

## API Additions (Non-Breaking)

### 1. Fluent API for Algorithm Chaining
```rust
graph.bfs(start)
    .filter(|node| node.degree() > 2)
    .map(|node| node.id())
    .collect::<Vec<_>>();
```

### 2. Graph Views and Projections
```rust
let subgraph = graph.view()
    .nodes(node_predicate)
    .edges(edge_predicate)
    .build();
```

### 3. Performance Hints
```rust
graph.with_hint(GraphHint::Dense)
    .with_hint(GraphHint::SmallWorld)
    .minimum_spanning_tree()
```

## Compatibility Guarantees

### Semantic Versioning Commitment
- 1.x.y: Only backward-compatible changes
- 2.0.0: May include breaking changes with migration path

### Stability Attributes
```rust
#[stable(feature = "graph_core", since = "1.0.0")]
pub struct Graph<N, E> { ... }

#[unstable(feature = "graph_neural", issue = "123")]
pub mod neural { ... }
```

## Testing Strategy for API Stability

1. **Compatibility Tests**: Ensure examples from 0.x continue to work
2. **Migration Tests**: Verify migration paths work correctly
3. **Performance Regression Tests**: API changes don't degrade performance
4. **Documentation Tests**: All examples in docs compile and run

## Recommendations Summary

### Must-Have for 1.0
1. ✅ Standardize return types for similar algorithms
2. ✅ Add deprecation warnings for changing APIs
3. ✅ Complete documentation for all public APIs
4. ✅ Ensure all examples compile and run

### Nice-to-Have for 1.0
1. ⭕ Fluent API for common operations
2. ⭕ Async variants for I/O operations
3. ⭕ Performance hints system

### Post-1.0 Roadmap
1. 🔮 Generic node types (2.0)
2. 🔮 GPU acceleration APIs
3. 🔮 Distributed graph processing APIs

## Action Items

- [ ] Implement `CommunityResult` type
- [ ] Add deprecation warnings to `shortest_path`
- [ ] Create migration tool for automated updates
- [ ] Update all documentation with stable API examples
- [ ] Add stability attributes to all public items
- [ ] Create API changelog for 1.0

## Conclusion

The scirs2-graph API is well-designed and close to stability. With minor adjustments to ensure consistency and the addition of proper deprecation warnings, the library will provide a solid foundation for graph processing in Rust while maintaining a clear upgrade path for future enhancements.