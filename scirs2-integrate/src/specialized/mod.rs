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
// Advanced-performance financial computing exports
pub use finance::advanced_monte_carlo__engine::{
    AdvancedMonteCarloEngine, OptionPricingResult, QuantumInspiredRNG, VarianceReductionSuite,
};
pub use finance::realtime_risk__engine::{
    AlertSeverity, RealTimeRiskMonitor, RiskAlert, RiskAlertType, RiskDashboard, RiskSnapshot,
};
pub use fluid__dynamics::turbulence__models::{
    FluidState3D, LESolver, RANSModel, RANSSolver, RANSState, SGSModel,
};
pub use fluid__dynamics::{
    DealiasingStrategy, FluidBoundaryCondition, FluidState, NavierStokesParams, NavierStokesSolver,
    SpectralNavierStokesSolver,
};
// Advanced-performance fluid dynamics exports
pub use fluid__dynamics::advanced_gpu__acceleration::{AdvancedGPUKernel, GPUMemoryPool};
pub use fluid__dynamics::neural_adaptive__solver::{
    AdaptiveAlgorithmSelector, AlgorithmRecommendation, ProblemCharacteristics,
};
pub use fluid__dynamics::streaming__optimization::StreamingComputeManager;
// Enhanced multiphase flow exports
pub use fluid__dynamics::multiphase__flow::{
    InterfaceTrackingMethod, MultiphaseFlowSolver, MultiphaseFlowState, PhaseProperties,
};
pub use quantum::quantum__algorithms::{
    MultiBodyQuantumSolver, QuantumAnnealer, VariationalQuantumEigensolver,
};
pub use quantum::{
    GPUMultiBodyQuantumSolver, GPUQuantumSolver, HarmonicOscillator, HydrogenAtom, ParticleInBox,
    QuantumPotential, QuantumState, SchrodingerMethod, SchrodingerSolver,
};
// Quantum machine learning exports
pub use quantum::quantum__algorithms::{
    EntanglementPattern, QuantumFeatureMap, QuantumKernelParams, QuantumSVMModel,
    QuantumSupportVectorMachine,
};
// Enhanced quantum optimization exports
pub use quantum::QuantumAnnealingSolver;
// Enhanced financial modeling exports
pub use finance::exotic__options::{
    AveragingType, ExoticOption, ExoticOptionPricer, ExoticOptionType, PricingResult,
    RainbowPayoffType,
};
pub use finance::risk__management::{PortfolioRiskMetrics, RiskAnalyzer, StressScenario};
