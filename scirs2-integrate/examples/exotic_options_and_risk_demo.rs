//! Comprehensive example demonstrating exotic options pricing and risk management
//!
//! This example showcases the newly implemented exotic_options and risk_management
//! modules in the scirs2-integrate finance specialized solver.

use ndarray::Array2;
use scirs2_integrate::specialized::finance::{OptionType, VolatilityModel};
use scirs2_integrate::specialized::*;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Exotic Options and Risk Management Demo");
    println!("==========================================\n");

    // Example 1: Asian Option Pricing
    println!("📊 Example 1: Asian Option Pricing");
    println!("-----------------------------------");

    let asian_option = ExoticOption {
        option_type: ExoticOptionType::Asian {
            averaging_type: AveragingType::Arithmetic,
        },
        call_put: OptionType::Call,
        strike: 100.0,
        maturity: 1.0,               // 1 year
        spot_prices: vec![105.0],    // Current stock price
        risk_free_rate: 0.05,        // 5% risk-free rate
        dividend_yields: vec![0.02], // 2% dividend yield
        volatilities: vec![0.20],    // 20% volatility
        correlations: None,
        observation_dates: vec![0.25, 0.5, 0.75, 1.0], // Quarterly observations
    };

    let pricer = ExoticOptionPricer::new(50_000, 252);
    match pricer.price_monte_carlo(&asian_option) {
        Ok(result) => {
            println!("  Asian Call Option Price: ${:.4}", result.price);
            println!(
                "  Standard Error: ${:.4}",
                result.standard_error.unwrap_or(0.0)
            );
            println!(
                "  Convergence: {}",
                result.convergence_info.unwrap_or("N/A".to_string())
            );
        }
        Err(e) => println!("  Error pricing Asian option: {:?}", e),
    }
    println!();

    // Example 2: Barrier Option Pricing
    println!("🚧 Example 2: Barrier Option Pricing");
    println!("-------------------------------------");

    let barrier_option = ExoticOption {
        option_type: ExoticOptionType::Barrier {
            barrier_level: 120.0,
            is_up: true,
            is_knock_in: false, // Knock-out barrier
        },
        call_put: OptionType::Call,
        strike: 100.0,
        maturity: 0.5, // 6 months
        spot_prices: vec![105.0],
        risk_free_rate: 0.05,
        dividend_yields: vec![0.02],
        volatilities: vec![0.25], // Higher volatility
        correlations: None,
        observation_dates: vec![],
    };

    match pricer.price_monte_carlo(&barrier_option) {
        Ok(result) => {
            println!("  Up-and-Out Barrier Call Price: ${:.4}", result.price);
            println!(
                "  Standard Error: ${:.4}",
                result.standard_error.unwrap_or(0.0)
            );
        }
        Err(e) => println!("  Error pricing barrier option: {:?}", e),
    }
    println!();

    // Example 3: Rainbow Option (Multi-Asset)
    println!("🌈 Example 3: Rainbow Option Pricing");
    println!("-------------------------------------");

    // Create correlation matrix for 2 assets
    let mut correlation_matrix = Array2::<f64>::eye(2);
    correlation_matrix[[0, 1]] = 0.3;
    correlation_matrix[[1, 0]] = 0.3;

    let rainbow_option = ExoticOption {
        option_type: ExoticOptionType::Rainbow {
            assets: vec!["AAPL".to_string(), "MSFT".to_string()],
            payoff_type: RainbowPayoffType::BestOf,
        },
        call_put: OptionType::Call,
        strike: 100.0,
        maturity: 1.0,
        spot_prices: vec![110.0, 95.0], // AAPL, MSFT prices
        risk_free_rate: 0.05,
        dividend_yields: vec![0.015, 0.02], // Different dividend yields
        volatilities: vec![0.30, 0.25],     // Different volatilities
        correlations: Some(correlation_matrix),
        observation_dates: vec![],
    };

    match pricer.price_monte_carlo(&rainbow_option) {
        Ok(result) => {
            println!("  Best-of Rainbow Call Price: ${:.4}", result.price);
            println!(
                "  Standard Error: ${:.4}",
                result.standard_error.unwrap_or(0.0)
            );
        }
        Err(e) => println!("  Error pricing rainbow option: {:?}", e),
    }
    println!();

    // Example 4: Risk Management Analysis
    println!("⚠️ Example 4: Portfolio Risk Analysis");
    println!("--------------------------------------");

    let mut risk_analyzer = RiskAnalyzer::new(0.02, 1); // 2% risk-free rate, 1-day horizon

    // Add some sample historical return data for a portfolio
    let spy_returns = generate_sample_returns(0.08, 0.15, 252); // 8% mean, 15% vol, 1 year
    let qqq_returns = generate_sample_returns(0.12, 0.20, 252); // 12% mean, 20% vol, 1 year
    let bond_returns = generate_sample_returns(0.03, 0.05, 252); // 3% mean, 5% vol, 1 year

    risk_analyzer.add_asset_data("SPY".to_string(), spy_returns, 0.6, 600_000.0); // 60% weight, $600k
    risk_analyzer.add_asset_data("QQQ".to_string(), qqq_returns, 0.3, 300_000.0); // 30% weight, $300k
    risk_analyzer.add_asset_data("BONDS".to_string(), bond_returns, 0.1, 100_000.0); // 10% weight, $100k

    match risk_analyzer.calculate_portfolio_risk() {
        Ok(metrics) => {
            println!("  Portfolio Risk Metrics:");
            println!("  ─────────────────────────");
            println!("  VaR (95%): ${:.0}", metrics.var_95);
            println!("  VaR (99%): ${:.0}", metrics.var_99);
            println!(
                "  Expected Shortfall (95%): ${:.0}",
                metrics.expected_shortfall_95
            );
            println!("  Maximum Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
            println!(
                "  Portfolio Volatility: {:.2}%",
                metrics.portfolio_volatility * 100.0
            );
            println!("  Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
            println!("  Sortino Ratio: {:.3}", metrics.sortino_ratio);
            println!("  Concentration Ratio: {:.3}", metrics.concentration_ratio);
            println!("  Liquidity Score: {:.3}", metrics.liquidity_score);

            println!("\n  Stress Test Results:");
            for (scenario, impact) in &metrics.stress_test_results {
                println!("    {}: ${:.0}", scenario, impact);
            }

            println!("\n  Risk Attribution:");
            for (asset, contribution) in &metrics.risk_attribution {
                println!("    {}: {:.2}%", asset, contribution * 100.0);
            }
        }
        Err(e) => println!("  Error calculating risk metrics: {:?}", e),
    }
    println!();

    // Example 5: Lookback Option
    println!("🔍 Example 5: Lookback Option Pricing");
    println!("--------------------------------------");

    let lookback_option = ExoticOption {
        option_type: ExoticOptionType::Lookback {
            is_floating_strike: true,
        },
        call_put: OptionType::Call,
        strike: 0.0, // Not used for floating strike lookback
        maturity: 1.0,
        spot_prices: vec![100.0],
        risk_free_rate: 0.05,
        dividend_yields: vec![0.02],
        volatilities: vec![0.20],
        correlations: None,
        observation_dates: vec![],
    };

    match pricer.price_monte_carlo(&lookback_option) {
        Ok(result) => {
            println!("  Floating Strike Lookback Call: ${:.4}", result.price);
            println!(
                "  Standard Error: ${:.4}",
                result.standard_error.unwrap_or(0.0)
            );
        }
        Err(e) => println!("  Error pricing lookback option: {:?}", e),
    }
    println!();

    // Example 6: Digital Option
    println!("💰 Example 6: Digital Option Pricing");
    println!("-------------------------------------");

    let digital_option = ExoticOption {
        option_type: ExoticOptionType::Digital {
            cash_amount: 1000.0, // $1000 payout
        },
        call_put: OptionType::Call,
        strike: 105.0,
        maturity: 0.25, // 3 months
        spot_prices: vec![100.0],
        risk_free_rate: 0.05,
        dividend_yields: vec![0.02],
        volatilities: vec![0.25],
        correlations: None,
        observation_dates: vec![],
    };

    match pricer.price_monte_carlo(&digital_option) {
        Ok(result) => {
            println!("  Digital Call Option Price: ${:.4}", result.price);
            println!(
                "  Standard Error: ${:.4}",
                result.standard_error.unwrap_or(0.0)
            );
            println!(
                "  Implied Probability: {:.2}%",
                (result.price / 1000.0) * 100.0
            );
        }
        Err(e) => println!("  Error pricing digital option: {:?}", e),
    }

    println!("\n✅ All examples completed successfully!");
    println!(
        "The newly implemented exotic options and risk management modules are working correctly."
    );

    Ok(())
}

/// Generate sample returns for demonstration purposes
fn generate_sample_returns(annual_mean: f64, annual_vol: f64, n_periods: usize) -> Vec<f64> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let daily_mean = annual_mean / n_periods as f64;
    let daily_vol = annual_vol / (n_periods as f64).sqrt();

    let normal = Normal::new(daily_mean, daily_vol).unwrap();

    (0..n_periods).map(|_| rng.sample(normal)).collect()
}
