# scirs2-core TODO

Core utilities and foundation for the SciRS2 scientific computing library in Rust.

## 🔧 Current Issues & Quick Fixes

### High Priority - Bug Fixes
- [x] ✅ **FIXED**: Fixed failing performance grade test in benchmarking module
- [x] ✅ **FIXED**: Fixed warnings in cross_file_prefetch and zero_copy_streaming modules
- [x] ✅ **FIXED**: Resolve module resolution error with `data.rs` vs `data/mod.rs` conflict
- [x] ✅ **FIXED**: Fix remaining doc test compilation errors in tracing and versioning modules (all doc tests now passing)
- [x] ✅ **FIXED**: Address remaining segmentation fault in memory tests (fixed safety tracker and leak detection issues)
- [x] ✅ **FIXED**: Fix ignored tests in array protocol and serialization modules (5 of 6 tests enabled; 1 thread safety test has alternative)
- [x] ✅ **FIXED**: Update API documentation for recently refactored validation modules

### Medium Priority - Maintenance
- [x] ✅ **COMPLETED**: Standardize error handling patterns across all modules (created guide, implemented conversions)
- [x] ✅ **COMPLETED**: Review and update dependency versions in workspace (updated 11 dependencies)
- [x] ✅ **COMPLETED**: Add missing unit tests for edge cases in validation system
- [ ] Improve code coverage in GPU and memory management modules
- [ ] Update examples to use latest API patterns

## 🚀 Current Development Focus (Alpha 5)

### Data Validation System Enhancement
- [x] ✅ **COMPLETED**: Modular data validation architecture with separate concerns
- [x] ✅ **COMPLETED**: Complete implementation of all constraint types (Pattern, Custom, etc.)
- [x] ✅ **COMPLETED**: Implement validation rule composition and chaining with AND, OR, NOT, and IF-THEN constraints
- [ ] Add JSON Schema integration for complex validation rules
- [x] ✅ **COMPLETED**: Add validation performance benchmarks and optimization
- [ ] Create validation DSL for complex business rules

### Production Readiness
- [ ] **Security Audit**: Complete security review of all public APIs
- [ ] **Performance Optimization**: Profile and optimize hot paths
- [ ] **Error Recovery**: Enhance circuit breaker patterns and retry logic
- [ ] **Documentation**: Complete API documentation with examples
- [ ] **Testing**: Achieve 95%+ test coverage across all modules

### API Stability & Compatibility
- [ ] **Breaking Change Management**: Document and minimize API changes
- [ ] **Version Migration**: Create migration guides for API updates
- [ ] **Backward Compatibility**: Ensure older versions remain functional
- [ ] **Integration Testing**: Test with all dependent modules

## 📋 Feature Development Roadmap

### Short Term (Next 2-4 weeks)
1. **Complete Validation System**
   - [x] ✅ **COMPLETED**: Finish constraint implementations (Pattern, Custom, Temporal)
   - [x] ✅ **COMPLETED**: Add comprehensive validation examples
   - [ ] Performance optimization for large-scale validation

2. **Memory Management Enhancements**
   - [ ] Implement cross-device memory management (CPU/GPU/TPU)
   - [ ] Add memory leak detection and automated cleanup
   - [ ] Optimize memory allocation patterns for scientific workloads

3. **Error Handling Improvements**
   - [ ] Standardize error context across all modules
   - [ ] Add structured error reporting with machine-readable format
   - [ ] Implement error aggregation for batch operations

### Medium Term (1-2 months)
1. **Distributed Computing Support**
   - [ ] Multi-node computation framework
   - [ ] Network-aware task distribution
   - [ ] Resource management across compute clusters

2. **Advanced GPU Acceleration**
   - [ ] Tensor core acceleration for supported hardware
   - [ ] Automatic kernel tuning for different GPU architectures
   - [ ] Heterogeneous computing (CPU+GPU hybrid processing)

3. **Scientific Data Structures**
   - [ ] Enhanced masked arrays with statistical operations
   - [ ] Sparse matrix optimizations
   - [ ] Time series data structures with temporal indexing

### Long Term (3-6 months)
1. **JIT Compilation Integration**
   - [ ] LLVM-based JIT compiler integration
   - [ ] Runtime optimization based on data characteristics
   - [ ] Domain-specific language for scientific computing

2. **Cloud Computing Support**
   - [ ] Cloud storage integration (S3, GCS, Azure)
   - [ ] Serverless computing support
   - [ ] Auto-scaling for variable workloads

3. **Advanced Analytics**
   - [ ] Statistical computing primitives
   - [ ] Machine learning pipeline integration
   - [ ] Real-time data processing capabilities

## 🧪 Testing & Quality Assurance

### Current Test Status
- ✅ **Unit Tests**: 321+ passing (with enabled array protocol tests)
- ✅ **Doc Tests**: 98 passing, 0 ignored  
- ✅ **Integration Tests**: 9 passing, 1 ignored (thread safety test has alternative)
- ✅ **Build Status**: Clean build with no warnings
- ✅ **All Issues Fixed**: Previously failing memory tests now pass
- ✅ **Dependencies**: Updated to latest compatible versions (as of 2025-06-20)

### Testing Priorities
- [ ] **Property-Based Testing**: Add QuickCheck-style tests for mathematical properties
- [ ] **Stress Testing**: Large dataset processing and memory pressure tests
- [ ] **Cross-Platform Testing**: Validate on Windows, macOS, Linux, and WASM
- [ ] **Performance Regression Testing**: Automated benchmark tracking
- [ ] **Security Testing**: Fuzzing and vulnerability scanning

## 📚 Documentation & Examples

### High Priority Documentation
- [ ] **Migration Guide**: Update guides for validation module changes
- [ ] **Best Practices**: Performance optimization patterns
- [ ] **Integration Examples**: Real-world usage with other scirs2 modules
- [ ] **Troubleshooting Guide**: Common issues and solutions

### Examples Needed
- [ ] Scientific computing workflows end-to-end
- [ ] GPU acceleration setup and usage
- [ ] Memory-efficient processing for large datasets
- [ ] Distributed computing examples
- [ ] Error handling and recovery patterns

## 🔮 Future Enhancements

### Research & Experimental Features
- [ ] **Quantum Computing Integration**: Basic quantum circuit simulation
- [ ] **WebAssembly Support**: Browser-based scientific computing
- [ ] **Edge Computing**: Optimizations for resource-constrained devices
- [ ] **Federated Learning**: Distributed model training framework

### API Evolution
- [ ] **Const Generics**: Leverage advanced const generic features
- [ ] **async/await**: Async scientific computing primitives
- [ ] **no_std Support**: Embedded and kernel-space computing
- [ ] **SIMD Improvements**: Auto-vectorization and advanced SIMD patterns

## ⚠️ Known Issues & Limitations

### Current Limitations
- Some array protocol operations not fully implemented (marked as ignored tests)
- GPU memory management limited to basic allocation/deallocation
- Distributed computing only supports thread-based parallelism
- JIT compilation interface incomplete
- Memory mapping limited to read-only operations

### Planned Fixes
- GPU operations require proper backend selection and configuration
- Memory metrics may have performance overhead in tight loops
- Some validation constraints need regex feature for pattern matching
- Profiling tools need integration with external monitoring systems

## 🎯 Success Metrics

### Release Criteria (Alpha 6)
- [ ] Zero build warnings across all feature combinations
- [ ] 95%+ test coverage for core modules
- [ ] Complete API documentation with examples
- [ ] Performance benchmarks meet SciPy baseline
- [ ] Security audit completed with no critical issues

### Performance Targets
- [ ] Memory usage within 10% of NumPy for equivalent operations
- [ ] SIMD operations provide 2-4x speedup over scalar equivalents
- [ ] GPU operations show significant speedup for large arrays (>10k elements)
- [ ] Parallel processing scales linearly up to available cores

---

## 📝 Development Notes

### Recent Changes
- **2025-06-20**: Implemented dirty chunk tracking and persistence in out-of-core memory module with writer callbacks
- **2025-06-20**: Implemented convert_to_out_of_core utility function for copying in-memory arrays to out-of-core format
- **2025-06-20**: Implemented parallel utility functions (par_linspace, par_maximum, par_minimum) using Rayon
- **2025-06-20**: Fixed unimplemented!() macros in array protocol module (ArrayFunctionDecorator, ArrayProtocolFunction)
- **2025-06-20**: Created comprehensive validation examples demonstrating all validation features
- **2025-06-20**: Implemented ChunkedArray::map and ChunkedArray::par_map methods for memory-efficient array processing
- **2025-06-20**: Implemented all validation constraints (Temporal, Statistical, Shape) with comprehensive tests
- **2025-06-20**: Fixed all clippy warnings across codebase (achieved zero warnings again)
- **2025-06-20**: Updated workspace dependencies: once_cell 1.19.0→1.21.3, rustfft 6.2.0→6.4.0, regex 1.10→1.11, libc 0.2→0.2.174, bzip2 0.4→0.4.4, itertools 0.13→0.14, tempfile 3.9→3.15, memmap2 0.7.1→0.9.5, cranelift-* 0.118.0→0.120.0
- **2025-06-20**: Fixed all ignored tests: enabled 5 of 6 tests (2 serialization, 3 array protocol); 1 thread safety test has alternative
- **2025-06-20**: Updated validation module documentation to reflect composite constraints and performance features
- **2025-06-20**: Standardized error handling patterns across modules with conversions and guide
- **2025-06-20**: Fixed segmentation faults in memory tests (safety tracker and leak detection)
- **2025-06-20**: Fixed all doc test compilation errors in tracing and versioning modules
- **2025-06-20**: Created validation performance benchmarks for optimization tracking
- **2025-06-20**: Implemented validation rule composition with AND, OR, NOT, and IF-THEN constraints
- **2025-06-20**: Fixed performance grade test assertion error in benchmarking module
- **2025-06-20**: Fixed build warnings in memory management modules
- **2024-06-17**: Fixed module resolution conflict in validation/data module
- **2024-06-17**: Updated doc tests to resolve compilation errors
- **2024-06-17**: Reorganized TODO.md for better actionability and clarity

### Next Actions for Contributors
1. Pick an item from "Current Issues & Quick Fixes" for immediate impact
2. Review "Feature Development Roadmap" for larger contributions
3. Check ignored tests - many are good candidates for implementation
4. Update documentation when implementing new features

### Contributing Guidelines
- All new features must include comprehensive tests
- Performance-sensitive code should include benchmarks
- Breaking changes require RFC and migration path
- Documentation updates are required for all API changes

**Last Updated**: 2024-06-17  
**Status**: Active Development - Alpha 6 Phase