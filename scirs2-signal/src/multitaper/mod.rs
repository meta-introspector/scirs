//! Multitaper spectral estimation.
//!
//! This module provides functions for spectral analysis using multitaper methods,
//! which provide robust spectral estimates with reduced variance and bias compared
//! to conventional approaches. The implementation includes Discrete Prolate
//! Spheroidal Sequences (DPSS) tapers, also known as Slepian sequences.

// Import internal modules
mod adaptive;
pub mod enhanced;
mod ftest;
mod jackknife;
mod psd;
mod utils;
mod windows;

// Re-export public components
pub use adaptive::adaptive_psd;
pub use enhanced::{enhanced_pmtm, EnhancedMultitaperResult, MultitaperConfig};
pub use ftest::{harmonic_ftest, multitaper_ftest, multitaper_ftest_complex};
pub use jackknife::{
    cross_spectrum_jackknife, jackknife_confidence_intervals, weighted_jackknife,
};
pub use psd::{multitaper_spectrogram, pmtm};
pub use utils::{coherence, multitaper_filtfilt};
pub use windows::dpss;

// No direct imports needed for the module - submodules import their own dependencies
