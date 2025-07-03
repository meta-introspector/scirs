//! Recurrent layer implementations

pub mod lstm;
pub mod gru;
pub mod rnn;
pub mod bidirectional;

// Re-export main types
pub use lstm::LSTM;
pub use gru::GRU;
pub use rnn::RNN;
pub use bidirectional::BidirectionalRNN;

// Type aliases for compatibility
use ndarray::{Array, IxDyn};
use std::sync::{Arc, RwLock};

/// Type alias for LSTM state cache (hidden, cell)
pub type LstmStateCache<F> = Arc<RwLock<Option<(Array<F, IxDyn>, Array<F, IxDyn>)>>>;

/// Type alias for LSTM gate cache (input, forget, output, cell gates)
pub type LstmGateCache<F> = Arc<RwLock<Option<(Array<F, IxDyn>, Array<F, IxDyn>, Array<F, IxDyn>, Array<F, IxDyn>)>>>;

/// Type alias for GRU state cache
pub type GruStateCache<F> = Arc<RwLock<Option<Array<F, IxDyn>>>>;

/// Type alias for GRU gate cache (reset, update, new gates)
pub type GruGateCache<F> = Arc<RwLock<Option<(Array<F, IxDyn>, Array<F, IxDyn>, Array<F, IxDyn>)>>>;

/// Type alias for RNN state cache
pub type RnnStateCache<F> = Arc<RwLock<Option<Array<F, IxDyn>>>>;