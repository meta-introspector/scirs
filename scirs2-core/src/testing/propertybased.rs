//! Property-based testing utilities

use quickcheck::{Arbitrary, Gen, QuickCheck};

/// Property test configuration
pub struct PropertyTestConfig {
    pub tests: u64,
    pub max_size: usize,
}

impl Default for PropertyTestConfig {
    fn default() -> Self {
        Self {
            tests: 100,
            max_size: 100,
        }
    }
}

/// Run property-based tests
pub fn run_property_test<F, A>(property: F, config: PropertyTestConfig)
where
    F: Fn(A) -> bool,
    A: Arbitrary,
{
    QuickCheck::new()
        .tests(config.tests)
        .max_tests(config.tests as usize)
        .quickcheck(property as fn(A) -> bool);
}
