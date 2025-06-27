//! Specialized physics and engineering functions
//!
//! This module provides special functions commonly used in physics and engineering
//! applications that extend beyond the standard mathematical special functions.

#![allow(dead_code)]

use crate::error::{SpecialError, SpecialResult};
// use crate::{bessel::j0, gamma, precision::constants};
// use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use num_traits::Float;
use std::f64::consts::PI;

/// Planck's radiation law functions
pub mod blackbody {
    use super::*;

    /// Physical constants
    pub const PLANCK_CONSTANT: f64 = 6.626_070_15e-34; // J⋅s
    pub const BOLTZMANN_CONSTANT: f64 = 1.380_649e-23; // J/K
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0; // m/s
    pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8; // W⋅m⁻²⋅K⁻⁴

    /// Planck's law: spectral radiance of a black body
    ///
    /// B(ν, T) = (2hν³/c²) / (exp(hν/kT) - 1)
    ///
    /// # Arguments
    /// * `frequency` - Frequency in Hz
    /// * `temperature` - Temperature in Kelvin
    ///
    /// # Returns
    /// Spectral radiance in W⋅sr⁻¹⋅m⁻²⋅Hz⁻¹
    pub fn planck_law(frequency: f64, temperature: f64) -> SpecialResult<f64> {
        if temperature <= 0.0 {
            return Err(SpecialError::DomainError(
                "Temperature must be positive".to_string(),
            ));
        }
        if frequency < 0.0 {
            return Err(SpecialError::DomainError(
                "Frequency must be non-negative".to_string(),
            ));
        }

        if frequency == 0.0 {
            return Ok(0.0);
        }

        let h = PLANCK_CONSTANT;
        let k = BOLTZMANN_CONSTANT;
        let c = SPEED_OF_LIGHT;

        let h_nu = h * frequency;
        let k_t = k * temperature;
        let exp_term = (h_nu / k_t).exp();

        if exp_term.is_infinite() {
            // Low temperature limit
            return Ok(0.0);
        }

        let numerator = 2.0 * h * frequency.powi(3) / c.powi(2);
        let denominator = exp_term - 1.0;

        Ok(numerator / denominator)
    }

    /// Wien's displacement law: wavelength of maximum emission
    ///
    /// λ_max = b / T
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    ///
    /// # Returns
    /// Wavelength of maximum emission in meters
    pub fn wien_displacement(temperature: f64) -> SpecialResult<f64> {
        if temperature <= 0.0 {
            return Err(SpecialError::DomainError(
                "Temperature must be positive".to_string(),
            ));
        }

        const WIEN_CONSTANT: f64 = 2.897_771_955e-3; // m⋅K
        Ok(WIEN_CONSTANT / temperature)
    }

    /// Stefan-Boltzmann law: total radiated power per unit area
    ///
    /// j* = σT⁴
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    ///
    /// # Returns
    /// Power per unit area in W/m²
    pub fn stefan_boltzmann_law(temperature: f64) -> SpecialResult<f64> {
        if temperature < 0.0 {
            return Err(SpecialError::DomainError(
                "Temperature must be non-negative".to_string(),
            ));
        }

        Ok(STEFAN_BOLTZMANN * temperature.powi(4))
    }

    /// Rayleigh-Jeans law (classical limit for low frequencies)
    pub fn rayleigh_jeans(frequency: f64, temperature: f64) -> SpecialResult<f64> {
        if temperature <= 0.0 {
            return Err(SpecialError::DomainError(
                "Temperature must be positive".to_string(),
            ));
        }
        if frequency < 0.0 {
            return Err(SpecialError::DomainError(
                "Frequency must be non-negative".to_string(),
            ));
        }

        let k = BOLTZMANN_CONSTANT;
        let c = SPEED_OF_LIGHT;

        Ok(2.0 * frequency.powi(2) * k * temperature / c.powi(2))
    }
}

/// Antenna and radiation pattern functions
pub mod antenna {
    use super::*;

    /// Radiation pattern of a dipole antenna
    ///
    /// # Arguments
    /// * `theta` - Angle from the antenna axis in radians
    /// * `length` - Length of the dipole in wavelengths
    pub fn dipole_pattern(theta: f64, length: f64) -> f64 {
        if length == 0.0 {
            return 0.0;
        }

        let beta_l = PI * length;
        let numerator = ((beta_l * theta.cos()).cos() - beta_l.cos()).abs();
        let denominator = theta.sin().abs();

        if denominator < 1e-10 {
            // Handle singularity at theta = 0 or π
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Array factor for uniform linear array
    ///
    /// # Arguments
    /// * `n_elements` - Number of antenna elements
    /// * `spacing` - Element spacing in wavelengths
    /// * `theta` - Angle from broadside in radians
    /// * `phase_shift` - Progressive phase shift between elements in radians
    pub fn array_factor(
        n_elements: usize,
        spacing: f64,
        theta: f64,
        phase_shift: f64,
    ) -> Complex64 {
        if n_elements == 0 {
            return Complex64::new(0.0, 0.0);
        }

        let psi = 2.0 * PI * spacing * theta.sin() + phase_shift;
        let mut factor = Complex64::new(0.0, 0.0);

        for n in 0..n_elements {
            let phase = n as f64 * psi;
            factor += Complex64::new(phase.cos(), phase.sin());
        }

        factor / n_elements as f64
    }

    /// Friis transmission equation (in dB)
    ///
    /// # Arguments
    /// * `pt_dbm` - Transmitted power in dBm
    /// * `gt_db` - Transmitter antenna gain in dB
    /// * `gr_db` - Receiver antenna gain in dB
    /// * `frequency` - Frequency in Hz
    /// * `distance` - Distance in meters
    ///
    /// # Returns
    /// Received power in dBm
    pub fn friis_equation(
        pt_dbm: f64,
        gt_db: f64,
        gr_db: f64,
        frequency: f64,
        distance: f64,
    ) -> SpecialResult<f64> {
        if frequency <= 0.0 {
            return Err(SpecialError::DomainError(
                "Frequency must be positive".to_string(),
            ));
        }
        if distance <= 0.0 {
            return Err(SpecialError::DomainError(
                "Distance must be positive".to_string(),
            ));
        }

        let c = blackbody::SPEED_OF_LIGHT;
        let wavelength = c / frequency;
        let path_loss_db = 20.0 * ((4.0 * PI * distance / wavelength).log10());

        Ok(pt_dbm + gt_db + gr_db - path_loss_db)
    }
}

/// Acoustic and vibration functions
pub mod acoustics {
    use super::*;

    /// Speed of sound in air
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Celsius
    /// * `humidity` - Relative humidity (0-1)
    /// * `pressure` - Atmospheric pressure in Pa (default: 101325)
    pub fn speed_of_sound(temperature: f64, humidity: f64, pressure: f64) -> SpecialResult<f64> {
        if temperature < -273.15 {
            return Err(SpecialError::DomainError(
                "Temperature below absolute zero".to_string(),
            ));
        }
        if humidity < 0.0 || humidity > 1.0 {
            return Err(SpecialError::DomainError(
                "Humidity must be between 0 and 1".to_string(),
            ));
        }
        if pressure <= 0.0 {
            return Err(SpecialError::DomainError(
                "Pressure must be positive".to_string(),
            ));
        }

        // Simplified formula (more accurate formulas exist)
        let t_kelvin = temperature + 273.15;
        let base_speed = 331.3 * (t_kelvin / 273.15).sqrt();

        // Humidity correction (approximate)
        let humidity_correction = 0.6 * humidity;

        Ok(base_speed * (1.0 + humidity_correction))
    }

    /// Sound pressure level (SPL) in dB
    ///
    /// # Arguments
    /// * `pressure` - Sound pressure in Pa
    /// * `reference` - Reference pressure in Pa (default: 20 μPa for air)
    pub fn sound_pressure_level(pressure: f64, reference: f64) -> SpecialResult<f64> {
        if pressure <= 0.0 || reference <= 0.0 {
            return Err(SpecialError::DomainError(
                "Pressure values must be positive".to_string(),
            ));
        }

        Ok(20.0 * (pressure / reference).log10())
    }

    /// A-weighting filter response
    ///
    /// # Arguments
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    /// A-weighting response in dB
    pub fn a_weighting(frequency: f64) -> SpecialResult<f64> {
        if frequency <= 0.0 {
            return Err(SpecialError::DomainError(
                "Frequency must be positive".to_string(),
            ));
        }

        let f2 = frequency * frequency;
        let f4 = f2 * f2;

        let numerator = 12194.0f64.powi(2) * f4;
        let denominator = (f2 + 20.6f64.powi(2))
            * ((f2 + 107.7f64.powi(2)) * (f2 + 737.9f64.powi(2))).sqrt()
            * (f2 + 12194.0f64.powi(2));

        let ra = numerator / denominator;
        let a_weight = 20.0 * ra.log10() + 2.0;

        Ok(a_weight)
    }

    /// Helmholtz resonator frequency
    ///
    /// # Arguments
    /// * `volume` - Cavity volume in m³
    /// * `neck_length` - Neck length in m
    /// * `neck_area` - Neck cross-sectional area in m²
    /// * `temperature` - Temperature in Celsius
    pub fn helmholtz_frequency(
        volume: f64,
        neck_length: f64,
        neck_area: f64,
        temperature: f64,
    ) -> SpecialResult<f64> {
        if volume <= 0.0 || neck_length <= 0.0 || neck_area <= 0.0 {
            return Err(SpecialError::DomainError(
                "Dimensions must be positive".to_string(),
            ));
        }

        let c = speed_of_sound(temperature, 0.5, 101325.0)?;
        let effective_length = neck_length + 0.8 * neck_area.sqrt(); // End correction

        Ok(c / (2.0 * PI) * (neck_area / (volume * effective_length)).sqrt())
    }
}

/// Optics and photonics functions
pub mod optics {
    use super::*;

    /// Fresnel coefficients for reflection and transmission
    ///
    /// # Arguments
    /// * `n1` - Refractive index of medium 1
    /// * `n2` - Refractive index of medium 2
    /// * `theta_i` - Incident angle in radians
    /// * `polarization` - 's' for perpendicular, 'p' for parallel
    ///
    /// # Returns
    /// (reflection coefficient, transmission coefficient)
    pub fn fresnel_coefficients(
        n1: f64,
        n2: f64,
        theta_i: f64,
        polarization: char,
    ) -> SpecialResult<(Complex64, Complex64)> {
        if n1 <= 0.0 || n2 <= 0.0 {
            return Err(SpecialError::DomainError(
                "Refractive indices must be positive".to_string(),
            ));
        }

        let sin_theta_t = n1 / n2 * theta_i.sin();

        // Check for total internal reflection
        let cos_theta_t = if sin_theta_t.abs() > 1.0 {
            Complex64::new(0.0, (sin_theta_t * sin_theta_t - 1.0).sqrt())
        } else {
            Complex64::new((1.0 - sin_theta_t * sin_theta_t).sqrt(), 0.0)
        };

        let cos_theta_i = Complex64::new(theta_i.cos(), 0.0);
        let n1_c = Complex64::new(n1, 0.0);
        let n2_c = Complex64::new(n2, 0.0);

        match polarization {
            's' | 'S' => {
                let r_s = (n1_c * cos_theta_i - n2_c * cos_theta_t)
                    / (n1_c * cos_theta_i + n2_c * cos_theta_t);
                let t_s = 2.0 * n1_c * cos_theta_i / (n1_c * cos_theta_i + n2_c * cos_theta_t);
                Ok((r_s, t_s))
            }
            'p' | 'P' => {
                let r_p = (n2_c * cos_theta_i - n1_c * cos_theta_t)
                    / (n2_c * cos_theta_i + n1_c * cos_theta_t);
                let t_p = 2.0 * n1_c * cos_theta_i / (n2_c * cos_theta_i + n1_c * cos_theta_t);
                Ok((r_p, t_p))
            }
            _ => Err(SpecialError::DomainError(
                "Polarization must be 's' or 'p'".to_string(),
            )),
        }
    }

    /// Gaussian beam waist
    ///
    /// # Arguments
    /// * `z` - Distance from beam waist
    /// * `w0` - Beam waist radius
    /// * `wavelength` - Wavelength
    pub fn gaussian_beam_radius(z: f64, w0: f64, wavelength: f64) -> SpecialResult<f64> {
        if w0 <= 0.0 || wavelength <= 0.0 {
            return Err(SpecialError::DomainError(
                "Beam waist and wavelength must be positive".to_string(),
            ));
        }

        let z_r = PI * w0 * w0 / wavelength; // Rayleigh range
        Ok(w0 * (1.0 + (z / z_r).powi(2)).sqrt())
    }

    /// Numerical aperture
    ///
    /// # Arguments
    /// * `n_core` - Core refractive index
    /// * `n_cladding` - Cladding refractive index
    pub fn numerical_aperture(n_core: f64, n_cladding: f64) -> SpecialResult<f64> {
        if n_core <= 0.0 || n_cladding <= 0.0 {
            return Err(SpecialError::DomainError(
                "Refractive indices must be positive".to_string(),
            ));
        }
        if n_core <= n_cladding {
            return Err(SpecialError::DomainError(
                "Core index must be greater than cladding index".to_string(),
            ));
        }

        Ok((n_core * n_core - n_cladding * n_cladding).sqrt())
    }

    /// Brewster's angle
    ///
    /// # Arguments
    /// * `n1` - Refractive index of medium 1
    /// * `n2` - Refractive index of medium 2
    ///
    /// # Returns
    /// Brewster's angle in radians
    pub fn brewster_angle(n1: f64, n2: f64) -> SpecialResult<f64> {
        if n1 <= 0.0 || n2 <= 0.0 {
            return Err(SpecialError::DomainError(
                "Refractive indices must be positive".to_string(),
            ));
        }

        Ok((n2 / n1).atan())
    }
}

/// Heat transfer and thermodynamics functions
pub mod thermal {
    use super::*;

    /// Thermal diffusion length
    ///
    /// # Arguments
    /// * `diffusivity` - Thermal diffusivity in m²/s
    /// * `frequency` - Modulation frequency in Hz
    pub fn thermal_diffusion_length(diffusivity: f64, frequency: f64) -> SpecialResult<f64> {
        if diffusivity <= 0.0 {
            return Err(SpecialError::DomainError(
                "Diffusivity must be positive".to_string(),
            ));
        }
        if frequency <= 0.0 {
            return Err(SpecialError::DomainError(
                "Frequency must be positive".to_string(),
            ));
        }

        Ok((diffusivity / (PI * frequency)).sqrt())
    }

    /// Biot number
    ///
    /// # Arguments
    /// * `h` - Heat transfer coefficient in W/(m²·K)
    /// * `l_c` - Characteristic length in m
    /// * `k` - Thermal conductivity in W/(m·K)
    pub fn biot_number(h: f64, l_c: f64, k: f64) -> SpecialResult<f64> {
        if h < 0.0 || l_c <= 0.0 || k <= 0.0 {
            return Err(SpecialError::DomainError(
                "Parameters must be non-negative (positive for l_c and k)".to_string(),
            ));
        }

        Ok(h * l_c / k)
    }

    /// Nusselt number for natural convection on vertical plate
    ///
    /// # Arguments
    /// * `rayleigh` - Rayleigh number
    pub fn nusselt_vertical_plate(rayleigh: f64) -> SpecialResult<f64> {
        if rayleigh < 0.0 {
            return Err(SpecialError::DomainError(
                "Rayleigh number must be non-negative".to_string(),
            ));
        }

        // Churchill-Chu correlation
        if rayleigh < 1e9 {
            // Laminar flow
            Ok(0.68 + 0.67 * rayleigh.powf(0.25) / (1.0 + (0.492 / 0.9).powf(9.0 / 16.0)).powf(4.0 / 9.0))
        } else {
            // Turbulent flow
            Ok(0.825 + 0.387 * rayleigh.powf(1.0 / 6.0) / (1.0 + (0.492 / 0.9).powf(9.0 / 16.0)).powf(8.0 / 27.0))
        }
    }

    /// View factor between parallel plates
    ///
    /// # Arguments
    /// * `width` - Width of the plates
    /// * `height` - Height of the plates
    /// * `distance` - Distance between plates
    pub fn view_factor_parallel_plates(width: f64, height: f64, distance: f64) -> SpecialResult<f64> {
        if width <= 0.0 || height <= 0.0 || distance <= 0.0 {
            return Err(SpecialError::DomainError(
                "Dimensions must be positive".to_string(),
            ));
        }

        let x = width / distance;
        let y = height / distance;
        
        let term1 = ((1.0 + x * x) * (1.0 + y * y) / (1.0 + x * x + y * y)).sqrt();
        let term2 = x * (1.0 + y * y).sqrt() * (x / (1.0 + y * y).sqrt()).atan();
        let term3 = y * (1.0 + x * x).sqrt() * (y / (1.0 + x * x).sqrt()).atan();
        let term4 = x * (x).atan();
        let term5 = y * (y).atan();

        Ok(2.0 / (PI * x * y) * (term1.ln() + term2 + term3 - term4 - term5))
    }
}

/// Semiconductor physics functions
pub mod semiconductor {
    use super::*;

    /// Intrinsic carrier concentration in silicon
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    pub fn intrinsic_carrier_concentration_si(temperature: f64) -> SpecialResult<f64> {
        if temperature <= 0.0 {
            return Err(SpecialError::DomainError(
                "Temperature must be positive".to_string(),
            ));
        }

        // Parameters for silicon
        let nc_300 = 2.86e19; // cm^-3 at 300K
        let nv_300 = 3.10e19; // cm^-3 at 300K
        let eg_0 = 1.17; // eV at 0K
        let alpha = 4.73e-4; // eV/K
        let beta = 636.0; // K

        let eg = eg_0 - alpha * temperature * temperature / (temperature + beta);
        let nc = nc_300 * (temperature / 300.0).powf(1.5);
        let nv = nv_300 * (temperature / 300.0).powf(1.5);

        let k_ev = blackbody::BOLTZMANN_CONSTANT / 1.602e-19; // Boltzmann constant in eV/K
        let ni_squared = nc * nv * (-eg / (k_ev * temperature)).exp();

        Ok(ni_squared.sqrt())
    }

    /// Fermi-Dirac distribution
    ///
    /// # Arguments
    /// * `energy` - Energy in eV
    /// * `fermi_level` - Fermi level in eV
    /// * `temperature` - Temperature in Kelvin
    pub fn fermi_dirac(energy: f64, fermi_level: f64, temperature: f64) -> SpecialResult<f64> {
        if temperature < 0.0 {
            return Err(SpecialError::DomainError(
                "Temperature must be non-negative".to_string(),
            ));
        }

        if temperature == 0.0 {
            // T=0 limit: step function
            Ok(if energy < fermi_level { 1.0 } else { 0.0 })
        } else {
            let k_ev = blackbody::BOLTZMANN_CONSTANT / 1.602e-19; // eV/K
            Ok(1.0 / (1.0 + ((energy - fermi_level) / (k_ev * temperature)).exp()))
        }
    }

    /// Debye length
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    /// * `carrier_concentration` - Carrier concentration in cm^-3
    /// * `permittivity` - Relative permittivity
    pub fn debye_length(
        temperature: f64,
        carrier_concentration: f64,
        permittivity: f64,
    ) -> SpecialResult<f64> {
        if temperature <= 0.0 || carrier_concentration <= 0.0 || permittivity <= 0.0 {
            return Err(SpecialError::DomainError(
                "All parameters must be positive".to_string(),
            ));
        }

        let epsilon_0 = 8.854e-14; // F/cm
        let q = 1.602e-19; // C
        let k = blackbody::BOLTZMANN_CONSTANT;

        Ok(((epsilon_0 * permittivity * k * temperature) / (q * q * carrier_concentration * 1e6)).sqrt())
    }
}

/// Plasma physics functions
pub mod plasma {
    use super::*;

    /// Plasma frequency
    ///
    /// # Arguments
    /// * `electron_density` - Electron density in m^-3
    pub fn plasma_frequency(electron_density: f64) -> SpecialResult<f64> {
        if electron_density < 0.0 {
            return Err(SpecialError::DomainError(
                "Electron density must be non-negative".to_string(),
            ));
        }

        const ELECTRON_CHARGE: f64 = 1.602e-19; // C
        const ELECTRON_MASS: f64 = 9.109e-31; // kg
        const EPSILON_0: f64 = 8.854e-12; // F/m

        Ok((electron_density * ELECTRON_CHARGE * ELECTRON_CHARGE
            / (EPSILON_0 * ELECTRON_MASS)).sqrt() / (2.0 * PI))
    }

    /// Debye shielding distance
    ///
    /// # Arguments
    /// * `temperature` - Electron temperature in Kelvin
    /// * `electron_density` - Electron density in m^-3
    pub fn debye_length_plasma(temperature: f64, electron_density: f64) -> SpecialResult<f64> {
        if temperature <= 0.0 || electron_density <= 0.0 {
            return Err(SpecialError::DomainError(
                "Temperature and density must be positive".to_string(),
            ));
        }

        const ELECTRON_CHARGE: f64 = 1.602e-19; // C
        const EPSILON_0: f64 = 8.854e-12; // F/m
        let k = blackbody::BOLTZMANN_CONSTANT;

        Ok((EPSILON_0 * k * temperature / (electron_density * ELECTRON_CHARGE * ELECTRON_CHARGE)).sqrt())
    }

    /// Cyclotron frequency
    ///
    /// # Arguments
    /// * `magnetic_field` - Magnetic field strength in Tesla
    /// * `particle_mass` - Particle mass in kg
    /// * `particle_charge` - Particle charge in Coulombs
    pub fn cyclotron_frequency(
        magnetic_field: f64,
        particle_mass: f64,
        particle_charge: f64,
    ) -> SpecialResult<f64> {
        if particle_mass <= 0.0 {
            return Err(SpecialError::DomainError(
                "Particle mass must be positive".to_string(),
            ));
        }

        Ok(particle_charge.abs() * magnetic_field / (2.0 * PI * particle_mass))
    }

    /// Alfvén velocity
    ///
    /// # Arguments
    /// * `magnetic_field` - Magnetic field strength in Tesla
    /// * `mass_density` - Mass density in kg/m³
    pub fn alfven_velocity(magnetic_field: f64, mass_density: f64) -> SpecialResult<f64> {
        if mass_density <= 0.0 {
            return Err(SpecialError::DomainError(
                "Mass density must be positive".to_string(),
            ));
        }

        const MU_0: f64 = 4.0 * PI * 1e-7; // H/m
        Ok(magnetic_field / (MU_0 * mass_density).sqrt())
    }
}

/// Quantum mechanics utility functions
pub mod quantum {
    use super::*;

    /// De Broglie wavelength
    ///
    /// # Arguments
    /// * `momentum` - Momentum in kg⋅m/s
    pub fn de_broglie_wavelength(momentum: f64) -> SpecialResult<f64> {
        if momentum <= 0.0 {
            return Err(SpecialError::DomainError(
                "Momentum must be positive".to_string(),
            ));
        }

        Ok(blackbody::PLANCK_CONSTANT / momentum)
    }

    /// Compton wavelength
    ///
    /// # Arguments
    /// * `mass` - Particle mass in kg
    pub fn compton_wavelength(mass: f64) -> SpecialResult<f64> {
        if mass <= 0.0 {
            return Err(SpecialError::DomainError(
                "Mass must be positive".to_string(),
            ));
        }

        let h = blackbody::PLANCK_CONSTANT;
        let c = blackbody::SPEED_OF_LIGHT;

        Ok(h / (mass * c))
    }

    /// Bohr radius for hydrogen-like atoms
    ///
    /// # Arguments
    /// * `z` - Atomic number
    pub fn bohr_radius(z: u32) -> SpecialResult<f64> {
        if z == 0 {
            return Err(SpecialError::DomainError(
                "Atomic number must be positive".to_string(),
            ));
        }

        const BOHR_RADIUS_H: f64 = 5.291_772_109e-11; // m
        Ok(BOHR_RADIUS_H / z as f64)
    }

    /// Rydberg wavelength
    ///
    /// # Arguments
    /// * `n_initial` - Initial principal quantum number
    /// * `n_final` - Final principal quantum number
    /// * `z` - Atomic number
    pub fn rydberg_wavelength(n_initial: u32, n_final: u32, z: u32) -> SpecialResult<f64> {
        if n_initial == 0 || n_final == 0 || z == 0 {
            return Err(SpecialError::DomainError(
                "Quantum numbers and atomic number must be positive".to_string(),
            ));
        }
        if n_initial <= n_final {
            return Err(SpecialError::DomainError(
                "Initial state must be higher than final state".to_string(),
            ));
        }

        const RYDBERG_CONSTANT: f64 = 1.097_373e7; // m^-1
        let z_sq = (z * z) as f64;
        let term = z_sq * RYDBERG_CONSTANT * (1.0 / (n_final * n_final) as f64 - 1.0 / (n_initial * n_initial) as f64);

        Ok(1.0 / term)
    }
}

/// Signal propagation and transmission line functions
pub mod transmission_lines {
    use super::*;

    /// Characteristic impedance of coaxial cable
    ///
    /// # Arguments
    /// * `outer_diameter` - Outer conductor inner diameter
    /// * `inner_diameter` - Inner conductor outer diameter
    /// * `permittivity` - Relative permittivity of dielectric
    pub fn coax_impedance(
        outer_diameter: f64,
        inner_diameter: f64,
        permittivity: f64,
    ) -> SpecialResult<f64> {
        if outer_diameter <= inner_diameter {
            return Err(SpecialError::DomainError(
                "Outer diameter must be greater than inner diameter".to_string(),
            ));
        }
        if inner_diameter <= 0.0 || permittivity <= 0.0 {
            return Err(SpecialError::DomainError(
                "Diameters and permittivity must be positive".to_string(),
            ));
        }

        Ok(60.0 / permittivity.sqrt() * (outer_diameter / inner_diameter).ln())
    }

    /// Skin depth
    ///
    /// # Arguments
    /// * `frequency` - Frequency in Hz
    /// * `conductivity` - Electrical conductivity in S/m
    /// * `permeability` - Relative permeability
    pub fn skin_depth(frequency: f64, conductivity: f64, permeability: f64) -> SpecialResult<f64> {
        if frequency <= 0.0 || conductivity <= 0.0 || permeability <= 0.0 {
            return Err(SpecialError::DomainError(
                "All parameters must be positive".to_string(),
            ));
        }

        const MU_0: f64 = 4.0 * PI * 1e-7; // H/m
        Ok(1.0 / (PI * frequency * conductivity * permeability * MU_0).sqrt())
    }

    /// Reflection coefficient
    ///
    /// # Arguments
    /// * `z_load` - Load impedance (complex)
    /// * `z_0` - Characteristic impedance (real)
    pub fn reflection_coefficient(z_load: Complex64, z_0: f64) -> SpecialResult<Complex64> {
        if z_0 <= 0.0 {
            return Err(SpecialError::DomainError(
                "Characteristic impedance must be positive".to_string(),
            ));
        }

        let z_0_complex = Complex64::new(z_0, 0.0);
        Ok((z_load - z_0_complex) / (z_load + z_0_complex))
    }

    /// VSWR (Voltage Standing Wave Ratio)
    ///
    /// # Arguments
    /// * `reflection_coeff` - Magnitude of reflection coefficient
    pub fn vswr(reflection_coeff: f64) -> SpecialResult<f64> {
        if reflection_coeff < 0.0 || reflection_coeff > 1.0 {
            return Err(SpecialError::DomainError(
                "Reflection coefficient magnitude must be between 0 and 1".to_string(),
            ));
        }

        if reflection_coeff == 1.0 {
            Ok(f64::INFINITY)
        } else {
            Ok((1.0 + reflection_coeff) / (1.0 - reflection_coeff))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_blackbody_functions() {
        // Test Wien's displacement law
        let wavelength = blackbody::wien_displacement(5778.0).unwrap(); // Sun's temperature
        assert_relative_eq!(wavelength, 5.016e-7, epsilon = 1e-9); // ~500 nm

        // Test Stefan-Boltzmann law
        let power = blackbody::stefan_boltzmann_law(5778.0).unwrap();
        assert_relative_eq!(power, 6.32e7, epsilon = 1e5); // ~63 MW/m²

        // Test Planck's law
        let radiance = blackbody::planck_law(6e14, 5778.0).unwrap(); // 500 nm
        assert!(radiance > 0.0);
    }

    #[test]
    fn test_antenna_functions() {
        // Test dipole pattern
        let pattern = antenna::dipole_pattern(PI / 2.0, 0.5);
        assert!(pattern > 0.0);

        // Test array factor
        let af = antenna::array_factor(4, 0.5, PI / 4.0, 0.0);
        assert!(af.norm() <= 1.0);

        // Test Friis equation
        let pr = antenna::friis_equation(0.0, 3.0, 3.0, 2.4e9, 100.0).unwrap();
        assert!(pr < 0.0); // Should have path loss
    }

    #[test]
    fn test_acoustics_functions() {
        // Test speed of sound
        let c = acoustics::speed_of_sound(20.0, 0.5, 101325.0).unwrap();
        assert_relative_eq!(c, 343.0, epsilon = 2.0);

        // Test SPL
        let spl = acoustics::sound_pressure_level(1.0, 20e-6).unwrap();
        assert_relative_eq!(spl, 94.0, epsilon = 0.1);

        // Test A-weighting
        let a_weight = acoustics::a_weighting(1000.0).unwrap();
        assert_relative_eq!(a_weight, 0.0, epsilon = 0.1); // 0 dB at 1 kHz
    }

    #[test]
    fn test_optics_functions() {
        // Test Brewster's angle for glass
        let brewster = optics::brewster_angle(1.0, 1.5).unwrap();
        assert_relative_eq!(brewster, 0.9827, epsilon = 1e-4);

        // Test numerical aperture
        let na = optics::numerical_aperture(1.5, 1.4).unwrap();
        assert_relative_eq!(na, 0.5385, epsilon = 1e-4);

        // Test Gaussian beam
        let w = optics::gaussian_beam_radius(1e-3, 1e-3, 633e-9).unwrap();
        assert!(w > 1e-3);
    }

    #[test]
    fn test_quantum_functions() {
        // Test de Broglie wavelength
        let lambda = quantum::de_broglie_wavelength(6.626e-24).unwrap();
        assert_relative_eq!(lambda, 1e-10, epsilon = 1e-12); // ~1 Angstrom

        // Test Bohr radius
        let a0 = quantum::bohr_radius(1).unwrap();
        assert_relative_eq!(a0, 5.2917e-11, epsilon = 1e-14);

        // Test Rydberg wavelength for H-alpha
        let wavelength = quantum::rydberg_wavelength(3, 2, 1).unwrap();
        assert_relative_eq!(wavelength, 656.3e-9, epsilon = 1e-10);
    }
}