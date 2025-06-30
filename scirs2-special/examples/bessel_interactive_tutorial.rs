//! Interactive Bessel Functions Tutorial
//!
//! This tutorial provides an interactive learning experience for Bessel functions,
//! combining mathematical theory with hands-on exploration.
//!
//! Run with: cargo run --example bessel_interactive_tutorial

use ndarray::Array1;
use scirs2_special::bessel::*;
use scirs2_special::{j0_zeros, j1_zeros};
use std::io::{self, Write};

fn main() {
    println!("🚀 Welcome to the Interactive Bessel Functions Tutorial!");
    println!("════════════════════════════════════════════════════════");
    println!();

    show_introduction();

    loop {
        show_menu();
        match get_user_choice() {
            1 => theory_overview(),
            2 => explore_bessel_j(),
            3 => explore_bessel_y(),
            4 => explore_modified_bessel(),
            5 => explore_spherical_bessel(),
            6 => practical_applications(),
            7 => interactive_plotting(),
            8 => quiz_section(),
            9 => {
                println!("Thank you for using the Bessel Functions Tutorial! 🎓");
                break;
            }
            _ => println!("Invalid choice. Please try again."),
        }
        println!();
        pause();
    }
}

fn show_introduction() {
    println!("📚 BESSEL FUNCTIONS: Mathematical Foundations");
    println!("─────────────────────────────────────────────");
    println!();
    println!("Bessel functions are solutions to Bessel's differential equation:");
    println!("x²(d²y/dx²) + x(dy/dx) + (x² - ν²)y = 0");
    println!();
    println!("They appear in many physical problems with cylindrical symmetry:");
    println!("• Heat conduction in cylinders");
    println!("• Vibrations of circular membranes");
    println!("• Electromagnetic waves in cylindrical waveguides");
    println!("• Quantum mechanics (radial part of wave functions)");
    println!();
    println!("There are several types of Bessel functions:");
    println!("• Jₙ(x): Bessel functions of the first kind");
    println!("• Yₙ(x): Bessel functions of the second kind (Neumann functions)");
    println!("• Iₙ(x): Modified Bessel functions of the first kind");
    println!("• Kₙ(x): Modified Bessel functions of the second kind");
    println!("• jₙ(x): Spherical Bessel functions of the first kind");
    println!("• yₙ(x): Spherical Bessel functions of the second kind");
    println!();
}

fn show_menu() {
    println!("🎯 Choose a topic to explore:");
    println!("1. 📖 Mathematical Theory Overview");
    println!("2. 🟢 Bessel Functions of First Kind (Jₙ)");
    println!("3. 🔴 Bessel Functions of Second Kind (Yₙ)");
    println!("4. 🟡 Modified Bessel Functions (Iₙ, Kₙ)");
    println!("5. 🔵 Spherical Bessel Functions");
    println!("6. 🔧 Practical Applications");
    println!("7. 📊 Interactive Plotting");
    println!("8. 🧠 Knowledge Quiz");
    println!("9. 🚪 Exit");
    print!("\nEnter your choice (1-9): ");
    io::stdout().flush().unwrap();
}

fn get_user_choice() -> i32 {
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");
    input.trim().parse().unwrap_or(0)
}

fn theory_overview() {
    println!("\n📖 MATHEMATICAL THEORY OF BESSEL FUNCTIONS");
    println!("══════════════════════════════════════════════");

    println!("\n🔬 Historical Context:");
    println!("Bessel functions were first studied by Daniel Bernoulli (1738) and");
    println!("later systematically developed by Friedrich Bessel (1824) while");
    println!("studying planetary motion.");

    println!("\n📐 The Bessel Differential Equation:");
    println!("x²(d²y/dx²) + x(dy/dx) + (x² - ν²)y = 0");
    println!();
    println!("where ν is called the 'order' of the Bessel function.");

    println!("\n🌊 Series Representations:");
    println!("Jₙ(x) = (x/2)ⁿ ∑[k=0 to ∞] [(-1)ᵏ/(k!(n+k)!)] * (x/2)²ᵏ");

    println!("\n🎯 Key Properties:");
    println!("• Jₙ(0) = δₙ₀ (Kronecker delta: 1 if n=0, 0 otherwise)");
    println!("• J₋ₙ(x) = (-1)ⁿJₙ(x) for integer n");
    println!("• Yₙ(x) → -∞ as x → 0⁺ for n ≥ 0");
    println!("• For large x: Jₙ(x) ≈ √(2/πx) cos(x - nπ/2 - π/4)");

    println!("\n🔢 Let's verify some properties with actual calculations:");

    // Demonstrate J₀(0) = 1
    let j0_at_zero = j0(0.0);
    println!("J₀(0) = {:.10} (should be 1.0)", j0_at_zero);

    // Demonstrate J₁(0) = 0
    let j1_at_zero = j1(0.0);
    println!("J₁(0) = {:.10} (should be 0.0)", j1_at_zero);

    // Show oscillatory behavior for large x
    let x_large = 20.0;
    let j0_large = j0(x_large);
    let asymptotic = (2.0 / (std::f64::consts::PI * x_large)).sqrt()
        * (x_large - std::f64::consts::PI / 4.0).cos();
    println!("\nFor large x = {}:", x_large);
    println!("J₀({}) = {:.6}", x_large, j0_large);
    println!("Asymptotic approximation = {:.6}", asymptotic);
    println!(
        "Relative error = {:.2}%",
        ((j0_large - asymptotic) / j0_large * 100.0).abs()
    );
}

fn explore_bessel_j() {
    println!("\n🟢 BESSEL FUNCTIONS OF THE FIRST KIND (Jₙ)");
    println!("═══════════════════════════════════════════════");

    println!("\nBessel functions Jₙ(x) are the solutions that are finite at x = 0.");
    println!("They oscillate with decreasing amplitude as x increases.");

    loop {
        println!("\nChoose an exploration:");
        println!("1. Calculate Jₙ(x) for specific values");
        println!("2. Find zeros of Bessel functions");
        println!("3. Explore recurrence relations");
        println!("4. Compare different orders");
        println!("5. Return to main menu");
        print!("Choice: ");
        io::stdout().flush().unwrap();

        match get_user_choice() {
            1 => calculate_bessel_j_values(),
            2 => explore_bessel_zeros(),
            3 => explore_recurrence_relations(),
            4 => compare_bessel_orders(),
            5 => break,
            _ => println!("Invalid choice."),
        }
    }
}

fn calculate_bessel_j_values() {
    println!("\n🧮 Calculate Jₙ(x) for specific values");
    println!("────────────────────────────────────────");

    print!("Enter the order n (0, 1, 2, ...): ");
    io::stdout().flush().unwrap();
    let n: usize = get_user_input().parse().unwrap_or(0);

    print!("Enter the value x: ");
    io::stdout().flush().unwrap();
    let x: f64 = get_user_input().parse().unwrap_or(1.0);

    let result = match n {
        0 => j0(x),
        1 => j1(x),
        _ => jn(n as i32, x),
    };

    println!("\nResult: J{}({}) = {:.10}", n, x, result);

    // Show some context
    println!("\nFor comparison:");
    println!("J₀({}) = {:.10}", x, j0(x));
    println!("J₁({}) = {:.10}", x, j1(x));
    if n != 0 && n != 1 {
        println!("J₂({}) = {:.10}", x, jn(2, x));
    }

    // Educational note about behavior
    if x < 1.0 {
        println!("\n📝 Note: For small x, Jₙ(x) ≈ (x/2)ⁿ/n! for n > 0");
        let approx = (x / 2.0).powi(n as i32) / factorial(n);
        if n > 0 {
            println!("Small-x approximation: {:.10}", approx);
            println!(
                "Relative error: {:.2}%",
                ((result - approx) / result * 100.0).abs()
            );
        }
    }
}

fn explore_bessel_zeros() {
    println!("\n🎯 Zeros of Bessel Functions");
    println!("────────────────────────────");

    println!("Bessel function zeros are important in physics and engineering.");
    println!("They appear in vibration problems, heat conduction, etc.");

    print!("Enter order n (0 or 1): ");
    io::stdout().flush().unwrap();
    let n: usize = get_user_input().parse().unwrap_or(0);

    print!("How many zeros to display? ");
    io::stdout().flush().unwrap();
    let num_zeros: usize = get_user_input().parse().unwrap_or(5).min(10);

    println!("\nFirst {} zeros of J{}(x):", num_zeros, n);
    println!("Zero #  |  Value");
    println!("────────|──────────");

    let zeros = match n {
        0 => j0_zeros(num_zeros),
        1 => j1_zeros(num_zeros),
        _ => {
            println!("Only n=0 and n=1 supported for zero finding in this demo.");
            return;
        }
    };

    for (i, zero) in zeros.iter().enumerate() {
        println!("{:6}  | {:8.6}", i + 1, zero);

        // Verify it's actually a zero
        let function_value = match n {
            0 => j0(*zero),
            1 => j1(*zero),
            _ => 0.0,
        };
        if function_value.abs() < 1e-10 {
            print!("  ✓");
        } else {
            print!("  (f({:.6}) = {:.2e})", zero, function_value);
        }
        println!();
    }

    // Educational application
    if n == 0 {
        println!("\n🔧 Application: Vibrating Circular Membrane");
        println!(
            "The first zero α₁ ≈ {:.4} determines the fundamental frequency:",
            zeros[0]
        );
        println!("f₁ = (α₁/2πa)√(T/σ) where a=radius, T=tension, σ=surface density");
    }
}

fn explore_recurrence_relations() {
    println!("\n🔄 Recurrence Relations");
    println!("──────────────────────");

    println!("Bessel functions satisfy several recurrence relations:");
    println!("1. Jₙ₋₁(x) + Jₙ₊₁(x) = (2n/x)Jₙ(x)");
    println!("2. Jₙ₋₁(x) - Jₙ₊₁(x) = 2J'ₙ(x)");
    println!("3. (d/dx)[xⁿJₙ(x)] = xⁿJₙ₋₁(x)");

    print!("Enter x value to test relations: ");
    io::stdout().flush().unwrap();
    let x: f64 = get_user_input().parse().unwrap_or(5.0);

    if x == 0.0 {
        println!("Cannot test at x = 0 due to division by zero.");
        return;
    }

    // Test first recurrence relation for n=1
    let j0_val = j0(x);
    let j1_val = j1(x);
    let j2_val = jn(2, x);

    let lhs = j0_val + j2_val;
    let rhs = (2.0 / x) * j1_val;

    println!("\nTesting: J₀(x) + J₂(x) = (2/x)J₁(x) at x = {}", x);
    println!("Left side:  J₀({}) + J₂({}) = {:.8}", x, x, lhs);
    println!("Right side: (2/{})J₁({}) = {:.8}", x, x, rhs);
    println!("Difference: {:.2e}", (lhs - rhs).abs());

    if (lhs - rhs).abs() < 1e-12 {
        println!("✓ Recurrence relation verified!");
    } else {
        println!("⚠ Small numerical difference (expected for finite precision)");
    }
}

fn compare_bessel_orders() {
    println!("\n📊 Comparing Different Orders");
    println!("─────────────────────────────");

    print!("Enter x value: ");
    io::stdout().flush().unwrap();
    let x: f64 = get_user_input().parse().unwrap_or(3.0);

    println!("\nBessel functions of different orders at x = {}:", x);
    println!("Order n |  Jₙ({})    |  Behavior", x);
    println!("────────|────────────|──────────");

    for n in 0..=5 {
        let val = match n {
            0 => j0(x),
            1 => j1(x),
            _ => jn(n, x),
        };

        let behavior = if n == 0 {
            "Starts at 1, first to oscillate"
        } else if n == 1 {
            "Starts at 0, derivative of J₀"
        } else {
            "Higher order, smaller amplitude"
        };

        println!("{:7} | {:10.6} | {}", n, val, behavior);
    }

    println!("\n📈 Observations:");
    println!("• Higher-order Bessel functions have smaller amplitudes");
    println!("• All orders oscillate with roughly the same period for large x");
    println!("• J₀ starts at 1, while Jₙ (n>0) start at 0");
}

fn explore_bessel_y() {
    println!("\n🔴 BESSEL FUNCTIONS OF THE SECOND KIND (Yₙ)");
    println!("═══════════════════════════════════════════════");

    println!("Bessel functions Yₙ(x) are the other linearly independent solutions");
    println!("to Bessel's differential equation. They have a singularity at x = 0.");

    print!("Enter x value (must be > 0): ");
    io::stdout().flush().unwrap();
    let x: f64 = get_user_input().parse().unwrap_or(1.0);

    if x <= 0.0 {
        println!("Error: Yₙ(x) is not defined for x ≤ 0");
        return;
    }

    println!("\nValues at x = {}:", x);
    println!("Y₀({}) = {:.6}", x, y0(x));
    println!("Y₁({}) = {:.6}", x, y1(x));
    println!("Y₂({}) = {:.6}", x, yn(2, x));

    if x < 1.0 {
        println!("\n📝 For small x:");
        println!("Y₀(x) ≈ (2/π)[ln(x/2) + γ] where γ is Euler's constant");
        println!("Y₁(x) ≈ -(2/πx)");

        let euler_gamma = 0.5772156649015329;
        let y0_approx = (2.0 / std::f64::consts::PI) * ((x / 2.0).ln() + euler_gamma);
        let y1_approx = -2.0 / (std::f64::consts::PI * x);

        println!("Y₀ approximation: {:.6}", y0_approx);
        println!("Y₁ approximation: {:.6}", y1_approx);
    }
}

fn explore_modified_bessel() {
    println!("\n🟡 MODIFIED BESSEL FUNCTIONS (Iₙ, Kₙ)");
    println!("═══════════════════════════════════════");

    println!("Modified Bessel functions are solutions to the modified equation:");
    println!("x²(d²y/dx²) + x(dy/dx) - (x² + ν²)y = 0");
    println!("They are related to regular Bessel functions by:");
    println!("Iₙ(x) = i⁻ⁿJₙ(ix) and Kₙ(x) = (π/2)i^(n+1)[Jₙ(ix) + iYₙ(ix)]");

    print!("Enter x value: ");
    io::stdout().flush().unwrap();
    let x: f64 = get_user_input().parse().unwrap_or(1.0);

    println!("\nModified Bessel functions at x = {}:", x);
    println!("I₀({}) = {:.6} (exponentially growing)", x, i0(x));
    println!("I₁({}) = {:.6} (exponentially growing)", x, i1(x));
    println!("K₀({}) = {:.6} (exponentially decaying)", x, k0(x));
    println!("K₁({}) = {:.6} (exponentially decaying)", x, k1(x));

    println!("\n🔧 Applications:");
    println!("• Heat conduction in cylindrical coordinates");
    println!("• Modified Helmholtz equation");
    println!("• Mathematical physics (especially diffusion)");
}

fn explore_spherical_bessel() {
    println!("\n🔵 SPHERICAL BESSEL FUNCTIONS");
    println!("═════════════════════════════");

    println!("Spherical Bessel functions appear in problems with spherical symmetry.");
    println!("They are related to regular Bessel functions by:");
    println!("jₙ(x) = √(π/2x) J_{{n+1/2}}(x)");
    println!("yₙ(x) = √(π/2x) Y_{{n+1/2}}(x)");

    print!("Enter x value: ");
    io::stdout().flush().unwrap();
    let x: f64 = get_user_input().parse().unwrap_or(1.0);

    print!("Enter maximum order n: ");
    io::stdout().flush().unwrap();
    let max_n: i32 = get_user_input().parse().unwrap_or(3).min(5);

    println!("\nSpherical Bessel functions at x = {}:", x);
    println!("Order |  jₙ(x)     |  yₙ(x)     | Elementary form");
    println!("------|------------|------------|----------------");

    for n in 0..=max_n {
        let jn_val = spherical_jn(n, x);
        let yn_val = spherical_yn(n, x);

        let elementary = match n {
            0 => format!("sin(x)/x, -cos(x)/x"),
            1 => format!("sin(x)/x² - cos(x)/x, -cos(x)/x² - sin(x)/x"),
            _ => "Complex expression".to_string(),
        };

        println!(
            "{:5} | {:10.6} | {:10.6} | {}",
            n, jn_val, yn_val, elementary
        );
    }

    // Verify elementary forms for small orders
    if x != 0.0 {
        println!("\n✓ Verification of elementary forms:");
        let j0_elementary = x.sin() / x;
        let y0_elementary = -x.cos() / x;
        println!(
            "j₀({}) = {:.6}, sin(x)/x = {:.6}",
            x,
            spherical_jn(0, x),
            j0_elementary
        );
        println!(
            "y₀({}) = {:.6}, -cos(x)/x = {:.6}",
            x,
            spherical_yn(0, x),
            y0_elementary
        );
    }
}

fn practical_applications() {
    println!("\n🔧 PRACTICAL APPLICATIONS OF BESSEL FUNCTIONS");
    println!("══════════════════════════════════════════════");

    println!("Choose an application to explore:");
    println!("1. 🥁 Vibrating Circular Membrane (Drum)");
    println!("2. 🌡️  Heat Conduction in a Cylinder");
    println!("3. 📡 Electromagnetic Waves in Circular Waveguide");
    println!("4. ⚛️  Quantum Mechanics: Particle in a Box (cylindrical)");
    println!("5. Return to main menu");

    print!("Choice: ");
    io::stdout().flush().unwrap();

    match get_user_choice() {
        1 => drum_vibration_demo(),
        2 => heat_conduction_demo(),
        3 => waveguide_demo(),
        4 => quantum_mechanics_demo(),
        5 => return,
        _ => println!("Invalid choice."),
    }
}

fn drum_vibration_demo() {
    println!("\n🥁 VIBRATING CIRCULAR MEMBRANE (DRUM)");
    println!("════════════════════════════════════");

    println!("A circular drum head vibrates in modes characterized by Bessel functions.");
    println!("The displacement u(r,θ,t) = J_m(k_mn*r) * [A*cos(mθ) + B*sin(mθ)] * cos(ω_mn*t)");
    println!("where k_mn is the nth zero of J_m divided by the drum radius.");

    print!("Enter drum radius (cm): ");
    io::stdout().flush().unwrap();
    let radius: f64 = get_user_input().parse().unwrap_or(10.0);

    print!("Enter tension (N/m): ");
    io::stdout().flush().unwrap();
    let tension: f64 = get_user_input().parse().unwrap_or(1000.0);

    print!("Enter surface density (kg/m²): ");
    io::stdout().flush().unwrap();
    let density: f64 = get_user_input().parse().unwrap_or(0.01);

    println!("\nVibrational modes for circular drum:");
    println!("Mode (m,n) | Frequency (Hz) | Description");
    println!("-----------|----------------|------------");

    // Calculate first few modes
    let zeros_j0 = j0_zeros(3);
    let zeros_j1 = j1_zeros(3);

    for (n, &zero) in zeros_j0.iter().enumerate() {
        let k = zero / (radius / 100.0); // Convert cm to m
        let frequency = k * (tension / density).sqrt() / (2.0 * std::f64::consts::PI);
        println!(
            "  (0,{})   |    {:8.1}    | Symmetric mode",
            n + 1,
            frequency
        );
    }

    for (n, &zero) in zeros_j1.iter().enumerate() {
        let k = zero / (radius / 100.0);
        let frequency = k * (tension / density).sqrt() / (2.0 * std::f64::consts::PI);
        println!(
            "  (1,{})   |    {:8.1}    | One nodal diameter",
            n + 1,
            frequency
        );
    }

    println!(
        "\n📝 The fundamental mode (0,1) has frequency {:.1} Hz",
        zeros_j0[0] / (radius / 100.0) * (tension / density).sqrt() / (2.0 * std::f64::consts::PI)
    );
}

fn heat_conduction_demo() {
    println!("\n🌡️ HEAT CONDUCTION IN A CYLINDER");
    println!("═══════════════════════════════════");

    println!("Temperature distribution in a cylinder with initial temperature T₀:");
    println!("T(r,t) = T₀ * ∑[n=1 to ∞] A_n * J₀(α_n*r/R) * exp(-α_n²*κ*t/R²)");
    println!("where α_n are zeros of J₀ and κ is thermal diffusivity.");

    print!("Enter cylinder radius R (cm): ");
    io::stdout().flush().unwrap();
    let radius: f64 = get_user_input().parse().unwrap_or(5.0);

    print!("Enter thermal diffusivity κ (cm²/s): ");
    io::stdout().flush().unwrap();
    let kappa: f64 = get_user_input().parse().unwrap_or(0.1);

    print!("Enter time t (seconds): ");
    io::stdout().flush().unwrap();
    let time: f64 = get_user_input().parse().unwrap_or(10.0);

    let zeros = j0_zeros(5);

    println!("\nTemperature decay at center (r=0) vs time:");
    println!("(Relative to initial temperature T₀)");
    println!("Time (s) | T(0,t)/T₀");
    println!("---------|----------");

    for t in [0.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
        if t > time + 100.0 {
            break;
        }

        let mut temp_ratio = 0.0;
        for (n, &zero) in zeros.iter().enumerate() {
            let coeff = 2.0 / (zero * j1(zero));
            let exponential = (-zero.powi(2) * kappa * t / radius.powi(2)).exp();
            temp_ratio += coeff * exponential;
        }

        println!("{:8.0} | {:8.4}", t, temp_ratio);
    }
}

fn waveguide_demo() {
    println!("\n📡 ELECTROMAGNETIC WAVES IN CIRCULAR WAVEGUIDE");
    println!("══════════════════════════════════════════════");

    println!("In a circular waveguide, electromagnetic modes are characterized by");
    println!("Bessel functions. The cutoff frequency depends on Bessel function zeros.");

    print!("Enter waveguide radius (cm): ");
    io::stdout().flush().unwrap();
    let radius: f64 = get_user_input().parse().unwrap_or(1.0);

    let c = 2.998e10; // Speed of light in cm/s

    println!("\nLowest cutoff frequencies:");
    println!("Mode    | Cutoff Freq (GHz) | Field Pattern");
    println!("--------|-------------------|---------------");

    // TE modes (use Bessel function zeros)
    let zeros_j0 = j0_zeros(2);
    let zeros_j1 = j1_zeros(3);

    // TE₀₁ mode (first zero of J₁)
    let fc_te01 = zeros_j1[0] * c / (2.0 * std::f64::consts::PI * radius) / 1e9;
    println!("TE₀₁    |      {:8.2}      | No angular variation", fc_te01);

    // TM₀₁ mode (first zero of J₀)
    let fc_tm01 = zeros_j0[0] * c / (2.0 * std::f64::consts::PI * radius) / 1e9;
    println!("TM₀₁    |      {:8.2}      | Longitudinal field", fc_tm01);

    // TE₁₁ mode (second zero of J₁)
    let fc_te11 = zeros_j1[1] * c / (2.0 * std::f64::consts::PI * radius) / 1e9;
    println!(
        "TE₁₁    |      {:8.2}      | One angular variation",
        fc_te11
    );

    println!(
        "\n📝 The TE₀₁ mode has the lowest cutoff frequency: {:.2} GHz",
        fc_te01
    );
}

fn quantum_mechanics_demo() {
    println!("\n⚛️ QUANTUM MECHANICS: PARTICLE IN CYLINDRICAL BOX");
    println!("═══════════════════════════════════════════════");

    println!("For a particle in a cylindrical box, the radial part of the");
    println!("wavefunction involves Bessel functions:");
    println!("ψ(r,φ,z) = A * J_m(k_r*r) * e^(imφ) * sin(nπz/L)");

    print!("Enter cylinder radius (Angstroms): ");
    io::stdout().flush().unwrap();
    let radius: f64 = get_user_input().parse().unwrap_or(10.0);

    print!("Enter cylinder height (Angstroms): ");
    io::stdout().flush().unwrap();
    let height: f64 = get_user_input().parse().unwrap_or(20.0);

    // Constants (in atomic units, approximately)
    let hbar: f64 = 1.0; // ℏ = 1 in atomic units
    let mass: f64 = 1.0; // electron mass = 1 in atomic units
    let bohr = 0.529; // Bohr radius in Angstroms

    println!("\nLowest energy levels (in Hartree):");
    println!("State (m,k,n) | Energy | Description");
    println!("--------------|--------|------------");

    let zeros_j0 = j0_zeros(3);
    let zeros_j1 = j1_zeros(2);

    let mut energies = Vec::new();

    // Ground state and some excited states
    for n_z in 1..=3 {
        for (k_idx, &zero) in zeros_j0.iter().enumerate() {
            let k_r = zero * bohr / radius;
            let k_z = n_z as f64 * std::f64::consts::PI * bohr / height;
            let energy = hbar.powi(2) / (2.0 * mass) * (k_r.powi(2) + k_z.powi(2));
            energies.push((
                format!("(0,{},{})", k_idx + 1, n_z),
                energy,
                "Symmetric radial",
            ));
        }

        for (k_idx, &zero) in zeros_j1.iter().enumerate() {
            let k_r = zero * bohr / radius;
            let k_z = n_z as f64 * std::f64::consts::PI * bohr / height;
            let energy = hbar.powi(2) / (2.0 * mass) * (k_r.powi(2) + k_z.powi(2));
            energies.push((
                format!("(1,{},{})", k_idx + 1, n_z),
                energy,
                "Angular variation",
            ));
        }
    }

    // Sort by energy and show first few
    energies.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (i, (state, energy, desc)) in energies.iter().take(8).enumerate() {
        println!("{:13} | {:6.4} | {}", state, energy, desc);
        if i == 0 {
            println!("              |        | <- Ground state");
        }
    }
}

fn interactive_plotting() {
    println!("\n📊 INTERACTIVE PLOTTING");
    println!("═══════════════════════");

    println!("Choose what to plot:");
    println!("1. Bessel functions J₀, J₁, J₂");
    println!("2. Modified Bessel functions I₀, K₀");
    println!("3. Spherical Bessel functions j₀, j₁, j₂");
    println!("4. Custom function comparison");

    print!("Choice: ");
    io::stdout().flush().unwrap();

    match get_user_choice() {
        1 => plot_regular_bessel(),
        2 => plot_modified_bessel(),
        3 => plot_spherical_bessel(),
        4 => plot_custom_comparison(),
        _ => println!("Invalid choice."),
    }
}

fn plot_regular_bessel() {
    println!("\n📈 Plotting Regular Bessel Functions");
    println!("─────────────────────────────────────");

    let x_max = 15.0;
    let n_points = 150;
    let x_values: Vec<f64> = Array1::linspace(0.01, x_max, n_points).to_vec();

    println!(
        "\nBessel Functions J₀(x), J₁(x), J₂(x) for x ∈ [0, {}]",
        x_max
    );
    println!("(Approximate ASCII plot)");
    println!();

    // Create ASCII plot
    let plot_height = 15;
    let plot_width = 80;

    // Calculate function values
    let j0_values: Vec<f64> = x_values.iter().map(|&x| j0(x)).collect();
    let j1_values: Vec<f64> = x_values.iter().map(|&x| j1(x)).collect();
    let j2_values: Vec<f64> = x_values.iter().map(|&x| jn(2, x)).collect();

    // Find min/max for scaling
    let y_min = -0.5;
    let y_max = 1.0;

    // Print plot
    for row in 0..plot_height {
        let y = y_max - (row as f64 / (plot_height - 1) as f64) * (y_max - y_min);

        if row == 0 || row == plot_height - 1 {
            print!("{:5.2} ", y);
        } else if (y - 0.0).abs() < 0.05 {
            print!(" 0.00 ");
        } else {
            print!("      ");
        }

        for col in 0..plot_width {
            let x_idx = (col as f64 / plot_width as f64 * (n_points - 1) as f64) as usize;
            if x_idx < j0_values.len() {
                let j0_diff = (j0_values[x_idx] - y).abs();
                let j1_diff = (j1_values[x_idx] - y).abs();
                let j2_diff = (j2_values[x_idx] - y).abs();

                if j0_diff < 0.05 {
                    print!("0");
                } else if j1_diff < 0.05 {
                    print!("1");
                } else if j2_diff < 0.05 {
                    print!("2");
                } else if (y - 0.0).abs() < 0.02 {
                    print!("-");
                } else {
                    print!(" ");
                }
            } else {
                print!(" ");
            }
        }
        println!();
    }

    println!(
        "      0    {}    {}    {}    {}",
        x_max / 4.0,
        x_max / 2.0,
        3.0 * x_max / 4.0,
        x_max
    );
    println!();
    println!("Legend: 0 = J₀(x), 1 = J₁(x), 2 = J₂(x)");

    // Show some key values
    println!("\nKey values:");
    for x in [1.0, 5.0, 10.0] {
        println!(
            "x = {}: J₀ = {:6.3}, J₁ = {:6.3}, J₂ = {:6.3}",
            x,
            j0(x),
            j1(x),
            jn(2, x)
        );
    }
}

fn plot_modified_bessel() {
    println!("\n📈 Plotting Modified Bessel Functions");
    println!("──────────────────────────────────────");

    // For modified Bessel functions, we need different ranges
    let x_max = 3.0; // I₀ grows exponentially
    let n_points = 60;
    let _x_values: Vec<f64> = Array1::linspace(0.01, x_max, n_points).to_vec();

    println!(
        "\nModified Bessel Functions I₀(x) and K₀(x) for x ∈ [0, {}]",
        x_max
    );
    println!("Note: I₀ grows exponentially, K₀ decays exponentially");

    // Show values at specific points
    println!("\nValues at selected points:");
    println!("x     | I₀(x)    | K₀(x)    | Behavior");
    println!("------|----------|----------|----------");

    for &x in [0.1, 0.5, 1.0, 2.0, 3.0].iter() {
        if x <= x_max {
            let i0_val = i0(x);
            let k0_val = k0(x);

            let behavior = if x < 0.5 {
                "I₀≈1, K₀≈-ln(x)"
            } else if x < 2.0 {
                "Transition region"
            } else {
                "I₀ grows, K₀ decays"
            };

            println!("{:5.1} | {:8.4} | {:8.4} | {}", x, i0_val, k0_val, behavior);
        }
    }

    println!("\n📝 Asymptotic behavior:");
    println!("For large x: I₀(x) ≈ e^x/√(2πx), K₀(x) ≈ √(π/2x) e^(-x)");
}

fn plot_spherical_bessel() {
    println!("\n📈 Plotting Spherical Bessel Functions");
    println!("───────────────────────────────────────");

    let x_max = 15.0;
    println!(
        "\nSpherical Bessel Functions j₀(x), j₁(x), j₂(x) for x ∈ [0, {}]",
        x_max
    );

    // Show relationship to elementary functions
    println!("\nElementary function forms:");
    println!("j₀(x) = sin(x)/x");
    println!("j₁(x) = sin(x)/x² - cos(x)/x");
    println!("j₂(x) = (3/x³ - 1/x)sin(x) - (3/x²)cos(x)");

    println!("\nValues at selected points:");
    println!("x     | j₀(x)    | j₁(x)    | j₂(x)    | sin(x)/x");
    println!("------|----------|----------|----------|----------");

    for &x in [1.0f64, 3.14159, 6.28, 9.42, 12.57].iter() {
        if x <= x_max {
            let j0_val = spherical_jn(0, x);
            let j1_val = spherical_jn(1, x);
            let j2_val = spherical_jn(2, x);
            let sinc_val = x.sin() / x;

            println!(
                "{:5.2} | {:8.4} | {:8.4} | {:8.4} | {:8.4}",
                x, j0_val, j1_val, j2_val, sinc_val
            );
        }
    }
}

fn plot_custom_comparison() {
    println!("\n📊 Custom Function Comparison");
    println!("─────────────────────────────");

    println!("Choose comparison:");
    println!("1. Regular vs Modified Bessel (same order)");
    println!("2. Bessel vs Spherical Bessel");
    println!("3. Different orders of same type");

    print!("Choice: ");
    io::stdout().flush().unwrap();

    match get_user_choice() {
        1 => {
            println!("\nComparing J₀(x) vs I₀(x):");
            for &x in [0.5, 1.0, 2.0, 3.0].iter() {
                println!(
                    "x = {}: J₀ = {:7.4}, I₀ = {:7.4} (ratio I₀/J₀ = {:6.2})",
                    x,
                    j0(x),
                    i0(x),
                    i0(x) / j0(x)
                );
            }
        }
        2 => {
            println!("\nComparing J₀(x) vs j₀(x) = sin(x)/x:");
            for &x in [1.0, 5.0, 10.0].iter() {
                let j0_reg = j0(x);
                let j0_sph = spherical_jn(0, x);
                println!(
                    "x = {}: J₀ = {:7.4}, j₀ = {:7.4} (ratio = {:6.3})",
                    x,
                    j0_reg,
                    j0_sph,
                    j0_sph / j0_reg
                );
            }
        }
        3 => {
            println!("\nComparing different orders at x = 5.0:");
            let x = 5.0;
            for n in 0..=4 {
                let val = match n {
                    0 => j0(x),
                    1 => j1(x),
                    _ => jn(n, x),
                };
                println!("J{}({}) = {:8.5}", n, x, val);
            }
        }
        _ => println!("Invalid choice."),
    }
}

fn quiz_section() {
    println!("\n🧠 BESSEL FUNCTIONS KNOWLEDGE QUIZ");
    println!("═════════════════════════════════════");

    let mut score = 0;
    let total_questions = 5;

    // Question 1
    println!("\nQuestion 1: What is J₀(0)?");
    println!("a) 0");
    println!("b) 1");
    println!("c) ∞");
    println!("d) undefined");
    print!("Your answer: ");
    io::stdout().flush().unwrap();

    let answer1 = get_user_input().trim().to_lowercase();
    if answer1 == "b" {
        println!("✓ Correct! J₀(0) = 1");
        score += 1;
    } else {
        println!("✗ Incorrect. J₀(0) = 1 because J₀ is the only Bessel function finite at origin with this value.");
    }

    // Question 2
    println!("\nQuestion 2: What happens to Yₙ(x) as x → 0⁺?");
    println!("a) Approaches 0");
    println!("b) Approaches 1");
    println!("c) Approaches -∞");
    println!("d) Oscillates");
    print!("Your answer: ");
    io::stdout().flush().unwrap();

    let answer2 = get_user_input().trim().to_lowercase();
    if answer2 == "c" {
        println!("✓ Correct! Yₙ(x) has a logarithmic singularity at x = 0");
        score += 1;
    } else {
        println!("✗ Incorrect. Yₙ(x) → -∞ as x → 0⁺ due to logarithmic singularity.");
    }

    // Question 3
    println!("\nQuestion 3: Modified Bessel functions Iₙ(x) are solutions to which equation?");
    println!("a) x²y'' + xy' + (x² - ν²)y = 0");
    println!("b) x²y'' + xy' - (x² + ν²)y = 0");
    println!("c) x²y'' + xy' + x²y = 0");
    println!("d) xy'' + y' + xy = 0");
    print!("Your answer: ");
    io::stdout().flush().unwrap();

    let answer3 = get_user_input().trim().to_lowercase();
    if answer3 == "b" {
        println!("✓ Correct! Note the minus sign before (x² + ν²)");
        score += 1;
    } else {
        println!(
            "✗ Incorrect. Modified Bessel equation has minus sign: x²y'' + xy' - (x² + ν²)y = 0"
        );
    }

    // Question 4
    println!("\nQuestion 4: The relationship j₀(x) = ? (spherical Bessel)");
    println!("a) J₀(x)");
    println!("b) sin(x)/x");
    println!("c) cos(x)/x");
    println!("d) x*J₁(x)");
    print!("Your answer: ");
    io::stdout().flush().unwrap();

    let answer4 = get_user_input().trim().to_lowercase();
    if answer4 == "b" {
        println!("✓ Correct! j₀(x) = sin(x)/x is the simplest spherical Bessel function");
        score += 1;
    } else {
        println!("✗ Incorrect. j₀(x) = sin(x)/x, which makes it very simple to compute.");
    }

    // Question 5
    println!("\nQuestion 5: In what physical problem do Bessel function zeros appear?");
    println!("a) Heat conduction in rectangular plate");
    println!("b) Vibrations of circular drum");
    println!("c) Simple harmonic oscillator");
    println!("d) Radioactive decay");
    print!("Your answer: ");
    io::stdout().flush().unwrap();

    let answer5 = get_user_input().trim().to_lowercase();
    if answer5 == "b" {
        println!("✓ Correct! Zeros determine the vibrational frequencies of circular membranes");
        score += 1;
    } else {
        println!("✗ Incorrect. Bessel zeros appear in circular drum vibrations, determining mode frequencies.");
    }

    // Final score
    println!("\n🏆 QUIZ RESULTS");
    println!("═══════════════");
    println!("Your score: {}/{}", score, total_questions);

    match score {
        5 => println!("🌟 Perfect! You're a Bessel function expert!"),
        4 => println!("🎉 Excellent! You have strong understanding of Bessel functions."),
        3 => println!("👍 Good job! You understand the basics well."),
        2 => println!("📚 Not bad! Review the theory and try again."),
        _ => println!("🤔 Consider reviewing the tutorial sections for better understanding."),
    }
}

// Helper functions
fn get_user_input() -> String {
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");
    input.trim().to_string()
}

fn pause() {
    print!("\nPress Enter to continue...");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|x| x as f64).product()
}
