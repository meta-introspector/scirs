# scirs2-core Module Fix Strategy

## Overview
3,151 issues found across 230 files. This document provides a systematic approach to fix them efficiently.

## Issue Breakdown by Type

### 1. Direct unwrap() calls: 1,910 occurrences
**Pattern**: `.unwrap()`
**Fix Strategy**:
```rust
// Before
let value = something.unwrap();

// After - Option 1: Propagate error
let value = something?;

// After - Option 2: Provide context
let value = something.map_err(|_| CoreError::InvalidArgument(
    ErrorContext::new("Failed to get value")
))?;

// After - Option 3: For tests only
let value = something.expect("Test assertion: value should exist");
```

### 2. Division without zero check: 1,027 occurrences
**Pattern**: `a / b`, `a /= b`
**Fix Strategy**:
```rust
// Before
let result = numerator / denominator;

// After
use crate::safe_ops::safe_divide;
let result = safe_divide(numerator, denominator)?;
```

### 3. Mathematical operations without validation: 183 occurrences
**Patterns**: `.sqrt()`, `.ln()`, `.log()`, `.powf()`
**Fix Strategy**:
```rust
// Before
let root = value.sqrt();
let logarithm = value.ln();

// After
use crate::safe_ops::{safe_sqrt, safe_log};
let root = safe_sqrt(value)?;
let logarithm = safe_log(value)?;
```

### 4. Array access patterns: 55 occurrences
**Pattern**: `array[index]`, `.get().unwrap()`
**Fix Strategy**:
```rust
// Before
let value = array[index];
let value = array.get(index).unwrap();

// After
let value = array.get(index)
    .ok_or_else(|| CoreError::IndexError(
        ErrorContext::new(format!("Index {} out of bounds", index))
    ))?;
```

## Priority Order (by criticality)

### Critical Path (Fix First):
1. **src/numeric/*.rs** - Core numerical operations
2. **src/validation/*.rs** - Input validation
3. **src/array_protocol/*.rs** - Array operations
4. **src/memory_efficient/*.rs** - Memory operations
5. **src/simd_ops.rs** - SIMD operations

### Secondary Priority:
6. **src/parallel_ops.rs** - Parallel processing
7. **src/gpu/*.rs** - GPU operations
8. **src/cache/*.rs** - Caching logic
9. **src/io/*.rs** - I/O operations
10. **src/types/*.rs** - Type conversions

### Lower Priority:
11. **benches/*.rs** - Benchmarks (use expect())
12. **tests/*.rs** - Tests (use expect())
13. **examples/*.rs** - Examples (can keep some unwrap for clarity)

## Automated Fix Patterns

### Pattern 1: Simple unwrap() replacement
```bash
# For test files - replace unwrap() with expect()
sed -i 's/\.unwrap()/\.expect("Test assertion failed")/g' tests/*.rs

# For Option::unwrap() to ?
sed -i 's/\.unwrap()/?/g' src/**/*.rs  # Then manually review
```

### Pattern 2: Division operations
```python
# Use the fix_unwraps.py script to identify all divisions
# Then systematically replace with safe_divide
```

### Pattern 3: Parse operations
```rust
// Common pattern
let num: f64 = str.parse().unwrap();

// Replace with
let num: f64 = str.parse()
    .map_err(|e| CoreError::ParseError(
        ErrorContext::new(format!("Failed to parse '{}': {}", str, e))
    ))?;
```

## Module-Specific Strategies

### src/numeric/
- Use safe_ops for ALL mathematical operations
- Add domain checks for special functions
- Validate all conversions

### src/array_protocol/
- Bounds check all array accesses
- Validate shapes before operations
- Handle GPU memory allocation failures

### src/validation/
- Already partially fixed
- Focus on remaining cast operations
- Ensure all validation returns proper errors

### src/memory_efficient/
- Handle mmap failures gracefully
- Check file permissions
- Validate memory bounds

## Implementation Steps

### Step 1: Create module-specific fix files
```bash
# Generate per-module reports
for module in numeric array_protocol validation memory_efficient simd_ops; do
    grep -A5 -B2 "src/$module" scirs2-core_unwrap_report.md > fixes_${module}.md
done
```

### Step 2: Fix critical mathematical operations
1. Replace all divisions with safe_divide
2. Replace all sqrt/log operations with safe versions
3. Add input validation where missing

### Step 3: Fix error propagation
1. Replace unwrap() with ? in library code
2. Add error context for better debugging
3. Use expect() with descriptive messages in tests

### Step 4: Validation and testing
1. Run tests after each module fix
2. Check for performance regressions
3. Ensure error messages are helpful

## Code Templates

### Template 1: Public API method
```rust
pub fn compute_something<T>(input: &[T]) -> CoreResult<T> 
where
    T: Float + Display,
{
    // Input validation
    if input.is_empty() {
        return Err(CoreError::InvalidArgument(
            ErrorContext::new("Input cannot be empty")
        ));
    }
    
    // Safe operations
    let sum = input.iter().sum();
    let count = T::from(input.len())
        .ok_or_else(|| CoreError::ConversionError(
            ErrorContext::new("Failed to convert length to numeric type")
        ))?;
    
    safe_divide(sum, count)
}
```

### Template 2: Internal computation
```rust
fn internal_calc(a: f64, b: f64) -> CoreResult<f64> {
    // Validate inputs
    check_finite(a, "parameter a")?;
    check_finite(b, "parameter b")?;
    
    // Safe computation
    let sqrt_a = safe_sqrt(a)?;
    let log_b = safe_log(b)?;
    
    safe_divide(sqrt_a, log_b)
}
```

## Progress Tracking

### Week 1 Goals:
- [ ] Fix all critical path modules (1-5)
- [ ] Reduce unwrap count by at least 1,000
- [ ] All mathematical operations use safe wrappers
- [ ] No panics in core numerical code

### Success Metrics:
- Zero unwrap() in public APIs
- All divisions protected against zero
- All mathematical operations validated
- Comprehensive error messages
- Tests pass without regression

## Notes
- Focus on correctness over performance initially
- Performance critical paths can be optimized later
- Document any intentional unwrap() usage
- Consider creating a scirs2-core/src/safe_conversions.rs module

---
Generated: 2025-06-27
Target: Fix 1,000+ issues by end of Week 1