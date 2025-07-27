//! Advanced Error Handling and Recovery Mechanisms Example
//!
//! This example demonstrates the comprehensive error handling framework in SciRS2-Core,
//! including recovery strategies, circuit breakers, async error handling, and diagnostics.

use scirs2_core::error::prelude::*;
use scirs2_core::error::{
    diagnostics::{diagnose_error, ErrorDiagnostics},
    recovery::{hints, CircuitBreaker, ErrorAggregator, RecoveryStrategy, RetryExecutor},
};
use std::time::Duration;

#[cfg(feature = "async")]
use scirs2_core::error::async_handling::{
    retry_with_exponential_backoff, with_timeout, AsyncCircuitBreaker, AsyncProgressTracker,
};

#[allow(dead_code)]
fn main() -> CoreResult<()> {
    println!("🚀 SciRS2 Advanced Error Handling Demo\n");

    // 1. Basic error handling with context
    basic_error_handling_demo()?;

    // 2. Recovery strategies with retry mechanisms
    retry_mechanisms_demo()?;

    // 3. Circuit breaker pattern for fault tolerance
    circuit_breaker_demo()?;

    // 4. Error aggregation for batch operations
    error_aggregation_demo()?;

    // 5. Advanced error diagnostics
    error_diagnostics_demo()?;

    // 6. Async error handling (if async feature is enabled)
    #[cfg(feature = "async")]
    {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { async_error_handling_demo().await })?;
    }

    println!("✅ All error handling demonstrations completed successfully!");
    Ok(())
}

/// Demonstrate basic error handling with rich context
#[allow(dead_code)]
fn basic_error_handling_demo() -> CoreResult<()> {
    println!("📋 1. Basic Error Handling with Context");

    // Create errors with automatic location tracking
    let domain_error = domain_error!("Input value must be positive");
    let computation_error = computation_error!("Matrix is singular", "solve_linear_system");

    println!("   Domain Error: {domain_error}");
    println!("   Computation Error: {computation_error}");

    // Convert errors to recoverable format
    let recoverable = RecoverableError::new(domain_error)
        .with_metadata("input_value", "-5.0")
        .with_metadata("expected_range", "> 0.0");

    println!("\n   📊 Recovery Report:");
    println!("{}", recoverable.recovery_report());

    println!("   ✅ Basic error handling demo completed\n");
    Ok(())
}

/// Demonstrate retry mechanisms with different backoff strategies
#[allow(dead_code)]
fn retry_mechanisms_demo() -> CoreResult<()> {
    println!("🔄 2. Retry Mechanisms with Backoff Strategies");

    // Exponential backoff strategy
    let exponential_strategy = RecoveryStrategy::ExponentialBackoff {
        max_attempts: 3,
        initial_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(2),
        multiplier: 2.0,
    };

    let executor = RetryExecutor::new(exponential_strategy);

    let mut attempt_count = 0;
    let result = executor.execute(|| {
        attempt_count += 1;
        println!("   🔧 Attempt {attempt_count} of simulated operation");

        if attempt_count < 3 {
            Err(CoreError::ComputationError(error_context!(
                "Temporary failure - network timeout"
            )))
        } else {
            println!("   ✅ Operation succeeded on attempt {attempt_count}");
            Ok("Operation completed successfully")
        }
    });

    match result {
        Ok(message) => println!("   📊 Result: {message}"),
        Err(e) => println!("   ❌ Final failure: {e}"),
    }

    // Linear backoff strategy
    println!("\n   🔄 Testing linear backoff strategy:");
    let linear_strategy = RecoveryStrategy::LinearBackoff {
        max_attempts: 2,
        delay: Duration::from_millis(50),
    };

    let linear_executor = RetryExecutor::new(linear_strategy);
    let linear_result = linear_executor.execute(|| {
        println!("   🔧 Linear backoff attempt");
        Ok("Linear retry succeeded")
    });

    println!("   📊 Linear backoff result: {linear_result:?}");
    println!("   ✅ Retry mechanisms demo completed\n");

    Ok(())
}

/// Demonstrate circuit breaker pattern for fault tolerance
#[allow(dead_code)]
fn circuit_breaker_demo() -> CoreResult<()> {
    println!("⚡ 3. Circuit Breaker Pattern for Fault Tolerance");

    let circuit_breaker = CircuitBreaker::new(
        2,                          // failure threshold
        Duration::from_millis(100), // timeout
        Duration::from_millis(500), // recovery timeout
    );

    println!(
        "   📊 Initial circuit breaker status: {}",
        circuit_breaker.status()
    );

    // Simulate failures to trigger circuit breaker
    for i in 1..=4 {
        println!("   🔧 Circuit breaker test attempt {i}");

        let result = circuit_breaker.execute(|| {
            if i <= 2 {
                Err(CoreError::ComputationError(error_context!(
                    "Simulated service failure"
                )))
            } else {
                Ok(format!("Success on attempt {i}"))
            }
        });

        match result {
            Ok(msg) => println!("   ✅ {msg}"),
            Err(e) => println!("   ❌ Failed: {e}"),
        }

        println!("   📊 Circuit status: {}", circuit_breaker.status());
    }

    println!("   ✅ Circuit breaker demo completed\n");
    Ok(())
}

/// Demonstrate error aggregation for batch operations
#[allow(dead_code)]
fn error_aggregation_demo() -> CoreResult<()> {
    println!("📦 4. Error Aggregation for Batch Operations");

    let mut aggregator = ErrorAggregator::with_max_errors(5);

    // Simulate batch processing with some failures
    let operations = vec![
        ("Operation A", true),
        ("Operation B", false),
        ("Operation C", true),
        ("Operation D", false),
        ("Operation E", true),
    ];

    let mut successful_operations = Vec::new();

    for (name, should_succeed) in operations {
        if should_succeed {
            println!("   ✅ {name} completed successfully");
            successful_operations.push(name);
        } else {
            println!("   ❌ {name} failed");
            let error = CoreError::ComputationError(error_context!(format!(
                "{} encountered an error",
                name
            )));
            aggregator.add_simple_error(error);
        }
    }

    println!("\n   📊 Batch Operation Summary:");
    println!("   Successful operations: {successful_operations:?}");
    println!("   Total errors collected: {}", aggregator.error_count());

    if aggregator.has_errors() {
        println!("\n   📋 Error Summary:");
        println!("{}", aggregator.summary());

        if let Some(most_severe) = aggregator.most_severe_error() {
            println!("\n   🔥 Most severe error recovery suggestions:");
            println!("{}", most_severe.recovery_report());
        }
    }

    println!("   ✅ Error aggregation demo completed\n");
    Ok(())
}

/// Demonstrate advanced error diagnostics and pattern analysis
#[allow(dead_code)]
fn error_diagnostics_demo() -> CoreResult<()> {
    println!("🔍 5. Advanced Error Diagnostics and Analysis");

    // Simulate different types of errors for diagnostics
    let errors = vec![
        CoreError::MemoryError(error_context!(
            "Out of memory during large matrix multiplication"
        )),
        CoreError::ConvergenceError(error_context!(
            "Algorithm failed to converge after 1000 iterations"
        )),
        CoreError::ShapeError(error_context!(
            "Matrix dimensions incompatible: (100, 50) vs (60, 80)"
        )),
        CoreError::DomainError(error_context!("Input contains NaN values")),
    ];

    for (i, error) in errors.iter().enumerate() {
        println!("   🔬 Analyzing Error {} - {}", i + 1, error);

        // Get comprehensive diagnostic report
        let diagnostics = diagnose_error(error);
        println!("   📊 Diagnostic Report:");
        println!("{}", diagnostics.generate_report());

        // Record error for pattern analysis
        ErrorDiagnostics::global().record_error(error, format!("demo_context_{}", i + 1));
    }

    // Demonstrate recovery hints
    println!("   💡 Recovery Hint Examples:");
    println!("{}", hints::check_inputs());
    println!("{}", hints::numerical_stability());
    println!("{}", hints::memory_optimization());
    println!("{}", hints::algorithm_selection());

    println!("   ✅ Error diagnostics demo completed\n");
    Ok(())
}

/// Demonstrate async error handling capabilities
#[cfg(feature = "async")]
async fn async_error_handling_demo() -> CoreResult<()> {
    println!("⏰ 6. Async Error Handling and Recovery");

    // Timeout handling
    println!("   ⏱️  Testing timeout handling:");
    let timeout_result = with_timeout(
        async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok::<&str, CoreError>("Should timeout")
        },
        Duration::from_millis(100),
    )
    .await;

    match timeout_result {
        Ok(_) => println!("   ❌ Unexpected success"),
        Err(e) => println!("   ✅ Expected timeout: {}", e),
    }

    // Async retry with exponential backoff
    println!("\n   🔄 Testing async retry with exponential backoff:");
    let async_attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let retry_result = retry_with_exponential_backoff(
        || {
            let attempts = async_attempts.clone();
            async move {
                let current_attempt =
                    attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                println!("   🔧 Async attempt {}", current_attempt);
                if current_attempt < 3 {
                    Err(CoreError::ComputationError(error_context!(
                        "Async temporary failure"
                    )))
                } else {
                    Ok("Async operation succeeded")
                }
            }
        },
        3,                          // max attempts
        Duration::from_millis(50),  // initial delay
        Duration::from_millis(500), // max delay
        1.5,                        // multiplier
    )
    .await;

    match retry_result {
        Ok(message) => println!("   ✅ Async retry result: {}", message),
        Err(e) => println!("   ❌ Async retry failed: {}", e),
    }

    // Progress tracking
    println!("\n   📈 Testing async progress tracking:");
    let tracker = AsyncProgressTracker::new(5);

    for _i in 1..=5 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        tracker.complete_step();
        println!("   📊 {}", tracker.progress_report());
    }

    // Async circuit breaker
    println!("\n   ⚡ Testing async circuit breaker:");
    let async_breaker = AsyncCircuitBreaker::new(
        2,                          // failure threshold
        Duration::from_millis(100), // timeout
        Duration::from_millis(300), // recovery timeout
    );

    for i in 1..=3 {
        let result = async_breaker
            .execute(|| async move {
                if i <= 2 {
                    Err(CoreError::ComputationError(error_context!(
                        "Async service failure"
                    )))
                } else {
                    Ok(format!("Async success on attempt {}", i))
                }
            })
            .await;

        match result {
            Ok(msg) => println!("   ✅ {}", msg),
            Err(e) => println!("   ❌ Async circuit breaker: {}", e),
        }
    }

    println!("   ✅ Async error handling demo completed\n");
    Ok(())
}

/// Demonstrate integration with real scientific computing scenarios
#[allow(dead_code)]
fn scientific_computing_scenario() -> CoreResult<()> {
    println!("🔬 Scientific Computing Error Handling Scenario");

    // Simulate a complex scientific computation with potential failures
    let matrix_size = 1000;
    let max_iterations = 100;

    // Setup retry strategy for numerical methods
    let scientific_retry = RecoveryStrategy::ExponentialBackoff {
        max_attempts: 5,
        initial_delay: Duration::from_millis(200),
        max_delay: Duration::from_secs(10),
        multiplier: 1.5,
    };

    let executor = RetryExecutor::new(scientific_retry);

    let result = executor.execute(|| {
        // Simulate scientific computation
        simulate_iterative_solver(matrix_size, max_iterations)
    });

    match result {
        Ok(solution) => {
            println!("   ✅ Scientific computation completed: {solution}");
        }
        Err(e) => {
            println!("   ❌ Scientific computation failed: {e}");

            // Generate comprehensive diagnostic report
            let diagnostics = diagnose_error(&e);
            println!("\n   📊 Scientific Error Analysis:");
            println!("{}", diagnostics.generate_report());
        }
    }

    Ok(())
}

/// Simulate an iterative solver that might fail
#[allow(dead_code)]
fn simulate_iterative_solver(iterations: usize) -> CoreResult<String> {
    // Simulate different failure modes
    use rand::Rng;
    let mut rng = rand::rng();
    let failure_mode = rng.gen_range(0..4);

    match failure_mode {
        0 => {
            // Memory error for large matrices
            if _matrix_size > 500 {
                Err(CoreError::MemoryError(error_context!(format!(
                    "Insufficient memory for {}x{} matrix"..matrix_size,
                    matrix_size
                ))))
            } else {
                Ok(format!(
                    "Linear system solved for {matrix_size}x{matrix_size} matrix"
                ))
            }
        }
        1 => {
            // Convergence error
            if max_iterations < 50 {
                Err(CoreError::ConvergenceError(error_context!(format!(
                    "Failed to converge after {} _iterations",
                    max_iterations
                ))))
            } else {
                Ok(format!(
                    "Converged after {} _iterations",
                    max_iterations - 10
                ))
            }
        }
        2 => {
            // Domain error for ill-conditioned matrices
            Err(CoreError::DomainError(error_context!(
                "Matrix is singular or nearly singular"
            )))
        }
        _ => {
            // Success case
            Ok(format!(
                "Successfully solved {}x{} system in {} _iterations",
                matrix_size,
                matrix_size,
                max_iterations / 2
            ))
        }
    }
}
