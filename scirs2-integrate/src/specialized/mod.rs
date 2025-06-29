//! Specialized solvers for domain-specific problems
//!
//! This module provides optimized solvers for specific scientific domains:
//! - Quantum mechanics (Schrödinger equation)
//! - Fluid dynamics (Navier-Stokes)
//! - Financial modeling (stochastic PDEs)

pub mod finance;
pub mod fluid_dynamics;
pub mod quantum;

pub use finance::{
    FinanceMethod, FinancialOption, Greeks, JumpProcess, OptionStyle, OptionType,
    StochasticPDESolver, VolatilityModel,
};
// Ultra-performance financial computing exports
pub use finance::ultra_monte_carlo_engine::{
    OptionPricingResult, QuantumInspiredRNG, UltraMonteCarloEngine, VarianceReductionSuite,
};
pub use finance::realtime_risk_engine::{
    AlertSeverity, RealTimeRiskMonitor, RiskAlert, RiskAlertType, RiskDashboard, RiskSnapshot,
};
pub use fluid_dynamics::turbulence_models::{
    FluidState3D, LESolver, RANSModel, RANSSolver, RANSState, SGSModel,
};
pub use fluid_dynamics::{
    DealiasingStrategy, FluidBoundaryCondition, FluidState, NavierStokesParams, 
    NavierStokesSolver, SpectralNavierStokesSolver,
};
// Ultra-performance fluid dynamics exports
pub use fluid_dynamics::ultra_gpu_acceleration::{
    GPUMemoryPool, UltraGPUKernel,
};
pub use fluid_dynamics::neural_adaptive_solver::{
    AdaptiveAlgorithmSelector, AlgorithmRecommendation, ProblemCharacteristics,
};
pub use fluid_dynamics::streaming_optimization::{
    StreamingComputeManager,
};
// Enhanced multiphase flow exports
pub use fluid_dynamics::multiphase_flow::{
    InterfaceTrackingMethod, MultiphaseFlowSolver, MultiphaseFlowState, PhaseProperties,
};
pub use quantum::quantum_algorithms::{
    MultiBodyQuantumSolver, QuantumAnnealer, VariationalQuantumEigensolver,
};
pub use quantum::{
    GPUMultiBodyQuantumSolver, GPUQuantumSolver, HarmonicOscillator, HydrogenAtom, ParticleInBox,
    QuantumPotential, QuantumState, SchrodingerMethod, SchrodingerSolver,
};
// Quantum machine learning exports
pub use quantum::quantum_algorithms::{
    EntanglementPattern, QuantumFeatureMap, QuantumKernelParams, 
    QuantumSVMModel, QuantumSupportVectorMachine,
};
// Enhanced quantum optimization exports
pub use quantum::{
    QuantumAnnealingSolver, VariationalQuantumEigensolver,
};
// Enhanced financial modeling exports
pub use finance::exotic_options::{
    AveragingType, ExoticOption, ExoticOptionPricer, ExoticOptionType, PricingResult, RainbowPayoffType,
};
pub use finance::risk_management::{
    PortfolioRiskMetrics, RiskAnalyzer, StressScenario,
};
