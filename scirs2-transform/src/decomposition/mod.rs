//! Matrix decomposition techniques
//!
//! This module provides various matrix decomposition algorithms that can be used
//! for feature extraction, data compression, and interpretable representations.

mod nmf;
mod dictionary_learning;

pub use self::nmf::NMF;
pub use self::dictionary_learning::DictionaryLearning;