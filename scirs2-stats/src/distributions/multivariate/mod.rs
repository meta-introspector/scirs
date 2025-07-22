//! Multivariate distribution functions
//!
//! This module provides functionality for multivariate statistical distributions.

pub mod dirichlet;
pub mod inverse_wishart;
pub mod multinomial;
pub mod multivariate_lognormal;
pub mod normal;
pub mod student_t;
pub mod wishart;

pub use dirichlet::Dirichlet;
pub use inverse__wishart::InverseWishart;
pub use multinomial::Multinomial;
pub use multivariate__lognormal::MultivariateLognormal;
pub use normal::MultivariateNormal;
pub use student__t::MultivariateT;
pub use wishart::Wishart;

// Re-export convenience functions
pub use dirichlet::dirichlet;
pub use inverse__wishart::inverse_wishart;
pub use multinomial::multinomial;
pub use multivariate__lognormal::multivariate_lognormal;
pub use normal::multivariate_normal;
pub use student__t::multivariate_t;
pub use wishart::wishart;
