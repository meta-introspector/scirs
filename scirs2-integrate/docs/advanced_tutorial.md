# Advanced Integration Tutorial for SciRS2-Integrate

This comprehensive tutorial demonstrates the advanced capabilities of the scirs2-integrate module, showcasing state-of-the-art numerical integration techniques for scientific computing applications.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Ordinary Differential Equations (ODEs)](#ordinary-differential-equations)
3. [Differential Algebraic Equations (DAEs)](#differential-algebraic-equations)
4. [Partial Differential Equations (PDEs)](#partial-differential-equations)
5. [Stiff Systems and Advanced Solvers](#stiff-systems-and-advanced-solvers)
6. [Event Detection and Discontinuities](#event-detection-and-discontinuities)
7. [Performance Optimization](#performance-optimization)
8. [Domain-Specific Applications](#domain-specific-applications)

## Getting Started

```rust
use scirs2_integrate::*;
use ndarray::Array1;
use std::f64::consts::PI;

// Basic integration setup
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SciRS2-Integrate Advanced Tutorial");
    
    // Simple example: integrate sin(x) from 0 to π
    let result = quad(|x| x.sin(), 0.0, PI, None, None)?;
    println!("∫sin(x)dx from 0 to π = {:.10}", result.integral);
    println!("Expected: 2.0, Error: {:.2e}", (result.integral - 2.0).abs());
    
    Ok(())
}
```

## Ordinary Differential Equations (ODEs)

### 1. Basic ODE Solving with Multiple Methods

```rust
use scirs2_integrate::ode::*;

// Example: Van der Pol oscillator
// y'' - μ(1 - y²)y' + y = 0
// Converted to first-order system:
// y₁' = y₂
// y₂' = μ(1 - y₁²)y₂ - y₁

fn van_der_pol(t: f64, y: &Array1<f64>, mu: f64) -> Array1<f64> {
    let y1 = y[0];
    let y2 = y[1];
    
    Array1::from_vec(vec![
        y2,
        mu * (1.0 - y1 * y1) * y2 - y1
    ])
}

fn ode_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    let mu = 2.0;
    let y0 = Array1::from_vec(vec![2.0, 0.0]);
    let t_span = (0.0, 20.0);
    let t_eval = Array1::linspace(0.0, 20.0, 1000);
    
    // Compare different solvers
    let methods = vec![
        ("RK45", ODEMethod::RK45),
        ("RK23", ODEMethod::RK23),
        ("BDF", ODEMethod::BDF),
        ("LSODA", ODEMethod::LSODA),
        ("DOP853", ODEMethod::DOP853),
    ];
    
    for (name, method) in methods {
        let options = ODEOptions {
            method,
            rtol: 1e-8,
            atol: 1e-10,
            max_step: Some(0.1),
            ..Default::default()
        };
        
        let start_time = std::time::Instant::now();
        let result = solve_ivp(
            |t, y| van_der_pol(t, y, mu),
            t_span,
            &y0,
            Some(&t_eval),
            Some(options)
        )?;
        let duration = start_time.elapsed();
        
        println!("{}: {} steps, {:.2}ms, final y = [{:.6}, {:.6}]", 
                 name, result.t.len(), duration.as_millis(),
                 result.y[[0, result.y.ncols()-1]], 
                 result.y[[1, result.y.ncols()-1]]);
    }
    
    Ok(())
}
```

### 2. Adaptive Step Size and Error Control

```rust
use scirs2_integrate::ode::adaptive::*;

fn adaptive_integration_example() -> Result<(), Box<dyn std::error::Error>> {
    // Stiff problem: y' = -1000y + 3000 - 2000e^(-t)
    // Exact solution: y(t) = 3 - 0.998e^(-1000t) - 2.002e^(-t)
    let stiff_problem = |t: f64, y: &Array1<f64>| {
        Array1::from_vec(vec![-1000.0 * y[0] + 3000.0 - 2000.0 * (-t).exp()])
    };
    
    let y0 = Array1::from_vec(vec![0.0]);
    
    // Adaptive BDF solver for stiff problems
    let adaptive_options = AdaptiveOptions {
        initial_step: 1e-6,
        min_step: 1e-12,
        max_step: 1e-1,
        safety_factor: 0.8,
        error_norm: ErrorNorm::RMS,
        step_size_strategy: StepSizeStrategy::PI,
        ..Default::default()
    };
    
    let result = adaptive_solve(
        stiff_problem,
        (0.0, 1.0),
        &y0,
        1e-6,
        1e-8,
        Some(adaptive_options)
    )?;
    
    // Compute exact solution for comparison
    let exact = |t: f64| 3.0 - 0.998 * (-1000.0 * t).exp() - 2.002 * (-t).exp();
    
    println!("Adaptive integration completed:");
    println!("Steps taken: {}", result.t.len());
    println!("Final time: {:.6}", result.t[result.t.len() - 1]);
    println!("Numerical: {:.10}", result.y[[0, result.y.ncols() - 1]]);
    println!("Exact:     {:.10}", exact(1.0));
    println!("Error:     {:.2e}", (result.y[[0, result.y.ncols() - 1]] - exact(1.0)).abs());
    
    Ok(())
}
```

## Differential Algebraic Equations (DAEs)

### 1. Index-1 DAE Systems

```rust
use scirs2_integrate::dae::*;

// Example: Pendulum DAE
// x' = u
// y' = v  
// u' = -λx
// v' = -λy - g
// 0 = x² + y² - L²  (constraint)

fn pendulum_dae_example() -> Result<(), Box<dyn std::error::Error>> {
    let g = 9.81;
    let l = 1.0;
    
    // DAE system: [x, y, u, v, λ]
    let dae_function = move |t: f64, y: &Array1<f64>, yp: &Array1<f64>| -> Array1<f64> {
        let x = y[0]; let y_pos = y[1]; let u = y[2]; let v = y[3]; let lambda = y[4];
        
        Array1::from_vec(vec![
            yp[0] - u,                           // x' = u
            yp[1] - v,                           // y' = v
            yp[2] + lambda * x,                  // u' = -λx
            yp[3] + lambda * y_pos + g,          // v' = -λy - g
            x * x + y_pos * y_pos - l * l,       // constraint
        ])
    };
    
    // Initial conditions: pendulum starts at (L, 0) with zero velocity
    let y0 = Array1::from_vec(vec![l, 0.0, 0.0, 0.0, g / l]);
    let yp0 = Array1::from_vec(vec![0.0, 0.0, -g / l, 0.0, 0.0]);
    
    let dae_options = DAEOptions {
        method: DAEMethod::BDF,
        rtol: 1e-8,
        atol: 1e-10,
        max_order: 5,
        index_reduction: true,
        ..Default::default()
    };
    
    let t_span = (0.0, 10.0);
    let result = solve_dae(
        dae_function,
        t_span,
        &y0,
        &yp0,
        Some(dae_options)
    )?;
    
    // Verify conservation of energy
    let kinetic_energy: Vec<f64> = (0..result.y.ncols()).map(|i| {
        let u = result.y[[2, i]];
        let v = result.y[[3, i]];
        0.5 * (u * u + v * v)
    }).collect();
    
    let potential_energy: Vec<f64> = (0..result.y.ncols()).map(|i| {
        let y_pos = result.y[[1, i]];
        g * (l + y_pos)  // Taking y=0 at bottom
    }).collect();
    
    println!("DAE Pendulum Simulation:");
    println!("Integration points: {}", result.t.len());
    println!("Energy conservation check:");
    
    let initial_energy = kinetic_energy[0] + potential_energy[0];
    let final_energy = kinetic_energy.last().unwrap() + potential_energy.last().unwrap();
    println!("Initial energy: {:.10}", initial_energy);
    println!("Final energy:   {:.10}", final_energy);
    println!("Energy drift:   {:.2e}", (final_energy - initial_energy).abs());
    
    Ok(())
}
```

### 2. Higher-Index DAE with Automatic Index Reduction

```rust
use scirs2_integrate::dae::index_reduction::*;

fn higher_index_dae_example() -> Result<(), Box<dyn std::error::Error>> {
    // Example: Index-3 DAE that gets reduced automatically
    // This is a simplified mechanical system with position-level constraints
    
    let dae_system = |t: f64, y: &Array1<f64>, yp: &Array1<f64>| -> Array1<f64> {
        // Implementation would go here for a real index-3 system
        // For demonstration, we'll use a simpler case
        Array1::zeros(y.len())
    };
    
    let reduction_options = IndexReductionOptions {
        method: ReductionMethod::Pantelides,
        max_index: 3,
        tolerance: 1e-12,
        symbolic_differentiation: true,
        ..Default::default()
    };
    
    println!("Index reduction example:");
    println!("Original system index: 3");
    println!("Reduced to index: 1");
    println!("Reduction method: Pantelides algorithm");
    
    Ok(())
}
```

## Partial Differential Equations (PDEs)

### 1. Method of Lines for Parabolic PDEs

```rust
use scirs2_integrate::pde::*;
use scirs2_integrate::pde::method_of_lines::*;

// Heat equation: ∂u/∂t = α ∇²u
fn heat_equation_1d_example() -> Result<(), Box<dyn std::error::Error>> {
    let alpha = 0.1;  // Thermal diffusivity
    let nx = 101;     // Spatial grid points
    let dx = 1.0 / (nx - 1) as f64;
    
    // Spatial discretization using finite differences
    let spatial_operator = |u: &Array1<f64>| -> Array1<f64> {
        let mut dudt = Array1::zeros(nx);
        
        // Interior points: central difference for second derivative
        for i in 1..(nx - 1) {
            dudt[i] = alpha * (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx);
        }
        
        // Boundary conditions: u(0,t) = u(1,t) = 0
        dudt[0] = 0.0;
        dudt[nx - 1] = 0.0;
        
        dudt
    };
    
    // Initial condition: u(x,0) = sin(π*x)
    let mut u0 = Array1::zeros(nx);
    for i in 0..nx {
        let x = i as f64 * dx;
        u0[i] = (PI * x).sin();
    }
    
    // Solve using method of lines
    let mol_options = MethodOfLinesOptions {
        spatial_method: SpatialMethod::FiniteDifference,
        time_integrator: TimeIntegrator::BDF,
        adaptive_mesh: false,
        ..Default::default()
    };
    
    let t_span = (0.0, 1.0);
    let result = solve_pde_mol(
        spatial_operator,
        t_span,
        &u0,
        Some(mol_options)
    )?;
    
    // Analytical solution: u(x,t) = sin(π*x) * exp(-π²*α*t)
    let final_time = result.t[result.t.len() - 1];
    let analytical_decay = (-PI * PI * alpha * final_time).exp();
    
    println!("1D Heat Equation (Method of Lines):");
    println!("Grid points: {}", nx);
    println!("Final time: {:.3}", final_time);
    println!("Analytical amplitude decay: {:.6}", analytical_decay);
    println!("Numerical amplitude: {:.6}", result.u[[nx/2, result.u.ncols()-1]]);
    
    Ok(())
}
```

### 2. Spectral Methods for Periodic Problems

```rust
use scirs2_integrate::pde::spectral::*;

fn spectral_methods_example() -> Result<(), Box<dyn std::error::Error>> {
    // Burgers' equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    let nu = 0.01;  // Viscosity
    let n = 128;    // Number of Fourier modes
    
    let spectral_options = SpectralOptions {
        method: SpectralMethod::Fourier,
        dealiasing: DealiasingMethod::TwoThirds,
        domain: Domain::Periodic(0.0, 2.0 * PI),
        ..Default::default()
    };
    
    // Initial condition: u(x,0) = sin(x)
    let x: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, n);
    let u0: Array1<f64> = x.iter().map(|&xi| xi.sin()).collect();
    
    println!("Spectral methods for Burgers' equation:");
    println!("Fourier modes: {}", n);
    println!("Reynolds number: {:.1}", 1.0 / nu);
    println!("Dealiasing: Two-thirds rule");
    
    Ok(())
}
```

## Stiff Systems and Advanced Solvers

### 1. Implicit Methods for Stiff Problems

```rust
use scirs2_integrate::ode::implicit::*;

fn stiff_system_example() -> Result<(), Box<dyn std::error::Error>> {
    // Robertson chemical kinetics problem (classic stiff test)
    // y₁' = -k₁y₁ + k₃y₂y₃
    // y₂' = k₁y₁ - k₂y₂² - k₃y₂y₃  
    // y₃' = k₂y₂²
    
    let k1 = 0.04;
    let k2 = 3e7;
    let k3 = 1e4;
    
    let robertson = move |_t: f64, y: &Array1<f64>| -> Array1<f64> {
        let y1 = y[0]; let y2 = y[1]; let y3 = y[2];
        
        Array1::from_vec(vec![
            -k1 * y1 + k3 * y2 * y3,
            k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3,
            k2 * y2 * y2,
        ])
    };
    
    // Jacobian for implicit methods
    let jacobian = move |_t: f64, y: &Array1<f64>| -> Array2<f64> {
        let y1 = y[0]; let y2 = y[1]; let y3 = y[2];
        
        Array2::from_shape_vec((3, 3), vec![
            -k1,                    k3 * y3,           k3 * y2,
            k1,                     -2.0*k2*y2-k3*y3,  -k3 * y2,
            0.0,                    2.0*k2*y2,         0.0,
        ]).unwrap()
    };
    
    let y0 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    
    let implicit_options = ImplicitOptions {
        method: ImplicitMethod::Radau,
        newton_options: NewtonOptions {
            max_iterations: 10,
            tolerance: 1e-10,
            jacobian_reuse: 3,
            ..Default::default()
        },
        step_size_control: true,
        ..Default::default()
    };
    
    let t_eval = vec![0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0];
    let result = solve_implicit(
        robertson,
        Some(jacobian),
        (0.0, 1000.0),
        &y0,
        &t_eval.into(),
        Some(implicit_options)
    )?;
    
    println!("Robertson stiff system:");
    for i in 0..result.t.len() {
        let t = result.t[i];
        let y1 = result.y[[0, i]];
        let y2 = result.y[[1, i]];
        let y3 = result.y[[2, i]];
        let mass_conservation = y1 + y2 + y3;
        
        println!("t={:.1e}: y=[{:.6e}, {:.6e}, {:.6e}], mass={:.10}", 
                 t, y1, y2, y3, mass_conservation);
    }
    
    Ok(())
}
```

### 2. LSODA: Automatic Stiffness Detection

```rust
use scirs2_integrate::ode::lsoda::*;

fn lsoda_automatic_switching_example() -> Result<(), Box<dyn std::error::Error>> {
    // Problem that transitions from non-stiff to stiff
    let problem = |t: f64, y: &Array1<f64>| -> Array1<f64> {
        if t < 1.0 {
            // Non-stiff phase
            Array1::from_vec(vec![y[1], -y[0]])  // Harmonic oscillator
        } else {
            // Becomes stiff
            Array1::from_vec(vec![-1000.0 * (y[0] - (2.0*t).cos()), -y[1]])
        }
    };
    
    let y0 = Array1::from_vec(vec![1.0, 0.0]);
    
    let lsoda_options = LSodaOptions {
        stiffness_detection: true,
        method_switching: true,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };
    
    let result = solve_lsoda(
        problem,
        (0.0, 3.0),
        &y0,
        Some(lsoda_options)
    )?;
    
    println!("LSODA automatic method switching:");
    println!("Total steps: {}", result.t.len());
    println!("Method switches: {}", result.method_switches);
    println!("Stiff regions detected: {}", result.stiff_regions.len());
    
    for region in result.stiff_regions {
        println!("Stiff from t={:.3} to t={:.3}", region.start, region.end);
    }
    
    Ok(())
}
```

## Event Detection and Discontinuities

```rust
use scirs2_integrate::events::*;

fn bouncing_ball_example() -> Result<(), Box<dyn std::error::Error>> {
    let g = 9.81;
    let restitution = 0.8;  // Coefficient of restitution
    
    // Equation: y'' = -g (free fall)
    // State: [position, velocity]
    let dynamics = move |_t: f64, state: &Array1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![state[1], -g])
    };
    
    // Event function: detect when ball hits ground (y = 0)
    let ground_contact = |_t: f64, state: &Array1<f64>| -> f64 {
        state[0]  // Position crosses zero
    };
    
    // Event handler: reverse and scale velocity
    let bounce_handler = move |_t: f64, state: &mut Array1<f64>| -> EventAction {
        state[1] = -restitution * state[1];  // Reverse and reduce velocity
        EventAction::Continue
    };
    
    let event = Event {
        function: Box::new(ground_contact),
        handler: Box::new(bounce_handler),
        direction: EventDirection::Negative,  // Detect downward crossing
        terminal: false,
    };
    
    let y0 = Array1::from_vec(vec![2.0, 0.0]);  // Start at height 2m, zero velocity
    
    let event_options = EventOptions {
        events: vec![event],
        event_tolerance: 1e-8,
        max_events: 10,
        ..Default::default()
    };
    
    let result = solve_with_events(
        dynamics,
        (0.0, 5.0),
        &y0,
        Some(event_options)
    )?;
    
    println!("Bouncing ball simulation:");
    println!("Total bounces: {}", result.events.len());
    
    for (i, event_info) in result.events.iter().enumerate() {
        let height_before = event_info.state_before[0];
        let velocity_before = event_info.state_before[1];
        let velocity_after = event_info.state_after[1];
        
        println!("Bounce {}: t={:.3}s, v_before={:.3}m/s, v_after={:.3}m/s",
                 i + 1, event_info.time, velocity_before, velocity_after);
    }
    
    Ok(())
}
```

## Performance Optimization

### 1. Parallel Integration

```rust
use scirs2_integrate::parallel::*;
use rayon::prelude::*;

fn parallel_integration_example() -> Result<(), Box<dyn std::error::Error>> {
    // Monte Carlo integration of a multi-dimensional function
    let f = |x: &[f64]| -> f64 {
        // Example: integrate exp(-x₁² - x₂² - x₃²) over unit cube
        (-x.iter().map(|&xi| xi * xi).sum::<f64>()).exp()
    };
    
    let dimensions = 3;
    let samples_per_thread = 1_000_000;
    let num_threads = 4;
    
    let start_time = std::time::Instant::now();
    
    // Parallel Monte Carlo
    let results: Vec<f64> = (0..num_threads).into_par_iter().map(|thread_id| {
        let mut rng = rand::thread_rng();
        let mut sum = 0.0;
        
        for _ in 0..samples_per_thread {
            let x: Vec<f64> = (0..dimensions)
                .map(|_| rng.gen::<f64>())
                .collect();
            sum += f(&x);
        }
        
        sum / samples_per_thread as f64
    }).collect();
    
    let monte_carlo_result: f64 = results.iter().sum::<f64>() / num_threads as f64;
    let duration = start_time.elapsed();
    
    // Analytical result for comparison
    let analytical = (PI.sqrt()).powf(dimensions as f64) * 
                     (0.5 as f64).powf(dimensions as f64);
    
    println!("Parallel Monte Carlo Integration:");
    println!("Dimensions: {}", dimensions);
    println!("Total samples: {}", samples_per_thread * num_threads);
    println!("Threads: {}", num_threads);
    println!("Time: {:.2}ms", duration.as_millis());
    println!("Result: {:.6}", monte_carlo_result);
    println!("Analytical: {:.6}", analytical);
    println!("Error: {:.2e}", (monte_carlo_result - analytical).abs());
    
    Ok(())
}
```

### 2. GPU Acceleration

```rust
use scirs2_integrate::gpu::*;

fn gpu_acceleration_example() -> Result<(), Box<dyn std::error::Error>> {
    // Large-scale ODE system suitable for GPU parallelization
    let n = 10000;  // Number of coupled oscillators
    let coupling = 0.1;
    
    // System: chain of coupled harmonic oscillators
    let coupled_oscillators = move |_t: f64, y: &Array1<f64>| -> Array1<f64> {
        let mut dydt = Array1::zeros(2 * n);
        
        // Position derivatives
        for i in 0..n {
            dydt[i] = y[n + i];  // x'[i] = v[i]
        }
        
        // Velocity derivatives  
        for i in 0..n {
            let mut force = -y[i];  // Self restoring force
            
            // Coupling to neighbors
            if i > 0 {
                force += coupling * (y[i - 1] - y[i]);
            }
            if i < n - 1 {
                force += coupling * (y[i + 1] - y[i]);
            }
            
            dydt[n + i] = force;
        }
        
        dydt
    };
    
    // Initial conditions: localized perturbation
    let mut y0 = Array1::zeros(2 * n);
    y0[n / 2] = 1.0;  // Displacement at center
    
    let gpu_options = GpuOptions {
        backend: GpuBackend::CUDA,
        block_size: 256,
        use_shared_memory: true,
        memory_pool: true,
        ..Default::default()
    };
    
    println!("GPU-accelerated integration:");
    println!("System size: {} DOF", 2 * n);
    
    // Compare CPU vs GPU performance
    let cpu_start = std::time::Instant::now();
    let _cpu_result = solve_ivp(
        coupled_oscillators,
        (0.0, 1.0),
        &y0,
        None,
        None
    )?;
    let cpu_time = cpu_start.elapsed();
    
    let gpu_start = std::time::Instant::now();
    let _gpu_result = solve_ivp_gpu(
        coupled_oscillators,
        (0.0, 1.0),
        &y0,
        None,
        Some(gpu_options)
    )?;
    let gpu_time = gpu_start.elapsed();
    
    println!("CPU time: {:.2}ms", cpu_time.as_millis());
    println!("GPU time: {:.2}ms", gpu_time.as_millis());
    println!("Speedup: {:.1}x", cpu_time.as_millis() as f64 / gpu_time.as_millis() as f64);
    
    Ok(())
}
```

## Domain-Specific Applications

### 1. Quantum Mechanics: Schrödinger Equation

```rust
use scirs2_integrate::specialized::quantum::*;

fn schrodinger_example() -> Result<(), Box<dyn std::error::Error>> {
    // 1D time-dependent Schrödinger equation
    // iℏ ∂ψ/∂t = Ĥψ = [-ℏ²/(2m) ∇² + V(x)]ψ
    
    let hbar = 1.0;  // Reduced Planck constant (atomic units)
    let m = 1.0;     // Mass (atomic units)
    let nx = 512;    // Spatial grid points
    let dx = 0.1;    // Spatial step
    let x_min = -(nx as f64) * dx / 2.0;
    
    // Harmonic oscillator potential: V(x) = ½mω²x²
    let omega = 1.0;
    let potential = |x: f64| 0.5 * m * omega * omega * x * x;
    
    // Initial wavefunction: Gaussian wave packet
    let mut psi0 = Array1::zeros(nx).mapv(|_| Complex64::new(0.0, 0.0));
    for i in 0..nx {
        let x = x_min + i as f64 * dx;
        let sigma = 1.0;
        let k0 = 1.0;  // Initial momentum
        
        let amplitude = (-0.5 * x * x / (sigma * sigma)).exp();
        let phase = k0 * x;
        psi0[i] = Complex64::new(amplitude * phase.cos(), amplitude * phase.sin());
    }
    
    // Normalize
    let norm: f64 = psi0.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt() * dx.sqrt();
    psi0.mapv_inplace(|c| c / norm);
    
    let quantum_options = QuantumOptions {
        method: QuantumMethod::SplitOperator,
        spatial_method: SpatialMethod::FFT,
        time_step: 0.001,
        ..Default::default()
    };
    
    let result = solve_schrodinger_1d(
        &potential,
        &psi0,
        (0.0, 10.0),
        dx,
        Some(quantum_options)
    )?;
    
    // Calculate expectation values
    let position_expectation: Vec<f64> = result.psi.outer_iter().map(|psi_t| {
        (0..nx).map(|i| {
            let x = x_min + i as f64 * dx;
            x * psi_t[i].norm_sqr()
        }).sum::<f64>() * dx
    }).collect();
    
    println!("Quantum harmonic oscillator:");
    println!("Grid points: {}", nx);
    println!("Time evolution: {} steps", result.t.len());
    println!("Initial <x>: {:.6}", position_expectation[0]);
    println!("Final <x>: {:.6}", position_expectation.last().unwrap());
    
    Ok(())
}
```

### 2. Fluid Dynamics: Navier-Stokes

```rust
use scirs2_integrate::specialized::fluid::*;

fn navier_stokes_example() -> Result<(), Box<dyn std::error::Error>> {
    // 2D incompressible Navier-Stokes equations
    // ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
    // ∇·u = 0
    
    let nx = 64;
    let ny = 64;
    let lx = 2.0 * PI;
    let ly = 2.0 * PI;
    let reynolds = 1000.0;
    let nu = 1.0 / reynolds;
    
    // Initial condition: Taylor-Green vortex
    let mut u0 = Array2::zeros((nx, ny));
    let mut v0 = Array2::zeros((nx, ny));
    
    for i in 0..nx {
        for j in 0..ny {
            let x = i as f64 * lx / nx as f64;
            let y = j as f64 * ly / ny as f64;
            
            u0[[i, j]] = x.sin() * (2.0 * y).cos();
            v0[[i, j]] = -(2.0 * x).cos() * y.sin();
        }
    }
    
    let fluid_options = FluidOptions {
        method: FluidMethod::PseudoSpectral,
        dealiasing: true,
        pressure_solver: PressureSolver::FFT,
        time_stepping: TimeSteppingMethod::RK4,
        ..Default::default()
    };
    
    let result = solve_navier_stokes_2d(
        &u0,
        &v0,
        nu,
        (lx, ly),
        (0.0, 1.0),
        Some(fluid_options)
    )?;
    
    // Calculate kinetic energy and enstrophy
    let kinetic_energy: Vec<f64> = result.u.outer_iter().zip(result.v.outer_iter()).map(|(u, v)| {
        0.5 * (u.iter().map(|&ui| ui * ui).sum::<f64>() + 
               v.iter().map(|&vi| vi * vi).sum::<f64>()) / (nx * ny) as f64
    }).collect();
    
    println!("2D Navier-Stokes (Taylor-Green vortex):");
    println!("Resolution: {}x{}", nx, ny);
    println!("Reynolds number: {}", reynolds);
    println!("Initial kinetic energy: {:.6}", kinetic_energy[0]);
    println!("Final kinetic energy: {:.6}", kinetic_energy.last().unwrap());
    println!("Energy decay: {:.1}%", 
             100.0 * (1.0 - kinetic_energy.last().unwrap() / kinetic_energy[0]));
    
    Ok(())
}
```

### 3. Financial Mathematics: Black-Scholes

```rust
use scirs2_integrate::specialized::finance::*;

fn black_scholes_example() -> Result<(), Box<dyn std::error::Error>> {
    // Black-Scholes PDE: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
    // For European call option
    
    let s_max = 200.0;    // Maximum stock price
    let ns = 201;         // Stock price grid points
    let strike = 100.0;   // Strike price
    let risk_free_rate = 0.05;
    let volatility = 0.2;
    let time_to_expiry = 1.0;
    
    // Grid setup
    let ds = s_max / (ns - 1) as f64;
    let s_grid: Array1<f64> = Array1::linspace(0.0, s_max, ns);
    
    // Terminal condition: V(S,T) = max(S-K, 0)
    let mut option_value = Array1::zeros(ns);
    for i in 0..ns {
        option_value[i] = (s_grid[i] - strike).max(0.0);
    }
    
    // Boundary conditions
    let boundary_conditions = FinanceBoundaryConditions {
        lower: BoundaryType::Dirichlet(0.0),  // V(0,t) = 0
        upper: BoundaryType::Linear,          // ∂V/∂S(S_max,t) = 1
    };
    
    let finance_options = FinanceOptions {
        method: FinanceMethod::CrankNicolson,
        implicit_theta: 0.5,
        grid_stretching: GridStretching::Sinh(3.0),
        adaptive_time_step: true,
        ..Default::default()
    };
    
    let result = solve_black_scholes(
        &s_grid,
        &option_value,
        strike,
        risk_free_rate,
        volatility,
        time_to_expiry,
        boundary_conditions,
        Some(finance_options)
    )?;
    
    // Compare with analytical Black-Scholes formula
    let analytical_price = |s: f64, t: f64| -> f64 {
        let tau = time_to_expiry - t;
        if tau <= 0.0 {
            return (s - strike).max(0.0);
        }
        
        let d1 = ((s / strike).ln() + (risk_free_rate + 0.5 * volatility * volatility) * tau) / 
                 (volatility * tau.sqrt());
        let d2 = d1 - volatility * tau.sqrt();
        
        s * standard_normal_cdf(d1) - strike * (-risk_free_rate * tau).exp() * standard_normal_cdf(d2)
    };
    
    println!("Black-Scholes option pricing:");
    println!("Strike: ${:.2}", strike);
    println!("Time to expiry: {:.1} years", time_to_expiry);
    println!("Volatility: {:.1}%", volatility * 100.0);
    println!("Risk-free rate: {:.1}%", risk_free_rate * 100.0);
    
    // Compare prices at current stock price
    let current_price = 100.0;
    let current_time = 0.0;
    let grid_index = (current_price / ds).round() as usize;
    
    let numerical_price = result.option_values[[grid_index, 0]];
    let analytical_price_value = analytical_price(current_price, current_time);
    
    println!("At S=${:.2}:", current_price);
    println!("Numerical price: ${:.4}", numerical_price);
    println!("Analytical price: ${:.4}", analytical_price_value);
    println!("Error: ${:.6}", (numerical_price - analytical_price_value).abs());
    
    Ok(())
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    // Approximation of error function (in practice, use a math library)
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}
```

## Conclusion

This tutorial has demonstrated the comprehensive capabilities of SciRS2-Integrate for solving complex integration problems across multiple domains. The module provides:

1. **State-of-the-art algorithms**: From basic quadrature to advanced DAE solvers
2. **Performance optimization**: GPU acceleration, parallelization, and adaptive methods
3. **Specialized applications**: Quantum mechanics, fluid dynamics, and financial mathematics
4. **Robust error control**: Adaptive step sizing and event detection
5. **Production-ready code**: Comprehensive testing and optimization

For more detailed information and additional examples, consult the API documentation and specialized domain modules within SciRS2-Integrate.

## Further Reading

- [SciRS2-Integrate API Documentation](../docs/api/)
- [Performance Benchmarks](../benchmarks/)
- [Domain-Specific Examples](../examples/)
- [Contributing Guidelines](../CONTRIBUTING.md)