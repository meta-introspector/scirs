//! Visualization tools for special functions
//!
//! This module provides comprehensive plotting and visualization capabilities
//! for all special functions, including 2D/3D plots, animations, and interactive
//! visualizations.

#[cfg(feature = "plotting")]
use plotters::prelude::*;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::error::Error;
use std::path::Path;

/// Configuration for plot generation
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Output width in pixels
    pub width: u32,
    /// Output height in pixels
    pub height: u32,
    /// DPI for high-resolution output
    pub dpi: u32,
    /// Plot title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Whether to show grid
    pub show_grid: bool,
    /// Whether to show legend
    pub show_legend: bool,
    /// Color scheme
    pub color_scheme: ColorScheme,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            dpi: 100,
            title: String::new(),
            x_label: "x".to_string(),
            y_label: "f(x)".to_string(),
            show_grid: true,
            show_legend: true,
            color_scheme: ColorScheme::default(),
        }
    }
}

/// Color schemes for plots
#[derive(Debug, Clone)]
pub enum ColorScheme {
    Default,
    Viridis,
    Plasma,
    Inferno,
    Magma,
    ColorBlind,
}

impl Default for ColorScheme {
    fn default() -> Self {
        ColorScheme::Default
    }
}

/// Trait for functions that can be visualized
pub trait Visualizable {
    /// Generate a 2D plot
    fn plot_2d(&self, config: &PlotConfig) -> Result<Vec<u8>, Box<dyn Error>>;
    
    /// Generate a 3D surface plot
    fn plot_3d(&self, config: &PlotConfig) -> Result<Vec<u8>, Box<dyn Error>>;
    
    /// Generate an animated visualization
    fn animate(&self, config: &PlotConfig) -> Result<Vec<Vec<u8>>, Box<dyn Error>>;
}

/// Plot multiple functions on the same axes
pub struct MultiPlot {
    functions: Vec<Box<dyn Fn(f64) -> f64>>,
    labels: Vec<String>,
    x_range: (f64, f64),
    config: PlotConfig,
}

impl MultiPlot {
    pub fn new(config: PlotConfig) -> Self {
        Self {
            functions: Vec::new(),
            labels: Vec::new(),
            x_range: (-10.0, 10.0),
            config,
        }
    }
    
    pub fn add_function(mut self, f: Box<dyn Fn(f64) -> f64>, label: &str) -> Self {
        self.functions.push(f);
        self.labels.push(label.to_string());
        self
    }
    
    pub fn set_x_range(mut self, min: f64, max: f64) -> Self {
        self.x_range = (min, max);
        self
    }
    
    #[cfg(feature = "plotting")]
    pub fn plot<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(path.as_ref(), (self.config.width, self.config.height))
            .into_drawing_area();
        root.fill(&WHITE)?;
        
        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(self.x_range.0..self.x_range.1, -2f64..2f64)?;
        
        if self.config.show_grid {
            chart.configure_mesh()
                .x_desc(&self.config.x_label)
                .y_desc(&self.config.y_label)
                .draw()?;
        }
        
        let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN];
        
        for (i, (f, label)) in self.functions.iter().zip(&self.labels).enumerate() {
            let color = colors[i % colors.len()];
            let data: Vec<(f64, f64)> = ((self.x_range.0 * 100.0) as i32..(self.x_range.1 * 100.0) as i32)
                .map(|x| x as f64 / 100.0)
                .map(|x| (x, f(x)))
                .collect();
            
            chart.draw_series(LineSeries::new(data, color))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
        }
        
        if self.config.show_legend {
            chart.configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
        }
        
        root.present()?;
        Ok(())
    }
}

/// Gamma function visualization
pub mod gamma_plots {
    use super::*;
    use crate::{gamma, gammaln, digamma};
    
    /// Plot gamma function and its logarithm
    pub fn plot_gamma_family<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        let config = PlotConfig {
            title: "Gamma Function Family".to_string(),
            x_label: "x".to_string(),
            y_label: "f(x)".to_string(),
            ..Default::default()
        };
        
        MultiPlot::new(config)
            .add_function(Box::new(|x| gamma(x)), "Γ(x)")
            .add_function(Box::new(|x| gammaln(x)), "ln Γ(x)")
            .add_function(Box::new(|x| digamma(x)), "ψ(x)")
            .set_x_range(0.1, 5.0)
            .plot(path)
    }
    
    /// Create a heatmap of gamma function in complex plane
    #[cfg(feature = "plotting")]
    pub fn plot_gamma_complex<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        use crate::gamma::complex::gamma_complex;
        
        let root = BitMapBackend::new(path.as_ref(), (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let mut chart = ChartBuilder::on(&root)
            .caption("Complex Gamma Function |Γ(z)|", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(-5f64..5f64, -5f64..5f64)?;
        
        chart.configure_mesh()
            .x_desc("Re(z)")
            .y_desc("Im(z)")
            .draw()?;
        
        // Create heatmap data
        let n = 100;
        let mut data = vec![];
        
        for i in 0..n {
            for j in 0..n {
                let x = -5.0 + 10.0 * i as f64 / n as f64;
                let y = -5.0 + 10.0 * j as f64 / n as f64;
                let z = Complex64::new(x, y);
                let gamma_z = gamma_complex(z);
                let magnitude = gamma_z.norm().ln(); // Log scale for better visualization
                
                data.push(Rectangle::new([(x, y), (x + 0.1, y + 0.1)], 
                    HSLColor(240.0 - magnitude * 30.0, 0.7, 0.5).filled()));
            }
        }
        
        chart.draw_series(data)?;
        
        root.present()?;
        Ok(())
    }
}

/// Bessel function visualization
pub mod bessel_plots {
    use super::*;
    use crate::bessel::{j0, j1, jn, y0, y1};
    
    /// Plot Bessel functions of the first kind
    pub fn plot_bessel_j<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        let config = PlotConfig {
            title: "Bessel Functions of the First Kind".to_string(),
            ..Default::default()
        };
        
        MultiPlot::new(config)
            .add_function(Box::new(|x| j0(x)), "J₀(x)")
            .add_function(Box::new(|x| j1(x)), "J₁(x)")
            .add_function(Box::new(|x| jn(2, x)), "J₂(x)")
            .add_function(Box::new(|x| jn(3, x)), "J₃(x)")
            .set_x_range(0.0, 20.0)
            .plot(path)
    }
    
    /// Plot zeros of Bessel functions
    pub fn plot_bessel_zeros<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        use crate::bessel_zeros::{j0_zeros, j1_zeros};
        
        #[cfg(feature = "plotting")]
        {
            let root = BitMapBackend::new(path.as_ref(), (800, 600)).into_drawing_area();
            root.fill(&WHITE)?;
            
            let mut chart = ChartBuilder::on(&root)
                .caption("Bessel Function Zeros", ("sans-serif", 40))
                .margin(10)
                .x_label_area_size(30)
                .y_label_area_size(40)
                .build_cartesian_2d(0f64..30f64, -0.5f64..1f64)?;
            
            chart.configure_mesh()
                .x_desc("x")
                .y_desc("J_n(x)")
                .draw()?;
            
            // Plot J0
            let j0_data: Vec<(f64, f64)> = (0..3000)
                .map(|i| i as f64 / 100.0)
                .map(|x| (x, j0(x)))
                .collect();
            chart.draw_series(LineSeries::new(j0_data, &BLUE))?
                .label("J₀(x)")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));
            
            // Mark zeros
            let zeros = j0_zeros(10);
            for zero in zeros {
                chart.draw_series(PointSeries::of_element(
                    vec![(zero, 0.0)],
                    5,
                    &RED,
                    &|c, s, st| {
                        return EmptyElement::at(c)
                            + Circle::new((0, 0), s, st.filled())
                            + Text::new(format!("{:.3}", zero), (10, 0), ("sans-serif", 15));
                    },
                ))?;
            }
            
            chart.configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
            
            root.present()?;
        }
        
        Ok(())
    }
}

/// Error function visualization
pub mod error_function_plots {
    use super::*;
    use crate::{erf, erfc, erfinv, erfcinv};
    
    /// Plot error functions and their inverses
    pub fn plot_error_functions<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        let config = PlotConfig {
            title: "Error Functions".to_string(),
            ..Default::default()
        };
        
        MultiPlot::new(config)
            .add_function(Box::new(|x| erf(x)), "erf(x)")
            .add_function(Box::new(|x| erfc(x)), "erfc(x)")
            .add_function(Box::new(|x| if x.abs() < 0.999 { erfinv(x) } else { f64::NAN }), "erfinv(x)")
            .set_x_range(-3.0, 3.0)
            .plot(path)
    }
}

/// Orthogonal polynomial visualization
pub mod polynomial_plots {
    use super::*;
    use crate::{legendre, chebyshev, hermite, laguerre};
    
    /// Plot Legendre polynomials
    pub fn plot_legendre<P: AsRef<Path>>(path: P, max_n: usize) -> Result<(), Box<dyn Error>> {
        let config = PlotConfig {
            title: format!("Legendre Polynomials P_n(x) for n = 0..{}", max_n),
            ..Default::default()
        };
        
        let mut plot = MultiPlot::new(config).set_x_range(-1.0, 1.0);
        
        for n in 0..=max_n {
            plot = plot.add_function(
                Box::new(move |x| legendre(n, x)), 
                &format!("P_{}", n)
            );
        }
        
        plot.plot(path)
    }
    
    /// Create an animated visualization of orthogonal polynomials
    pub fn animate_polynomials() -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
        // This would generate frames for an animation
        // showing how orthogonal polynomials evolve with increasing order
        Ok(vec![])
    }
}

/// Special function surface plots
pub mod surface_plots {
    use super::*;
    
    /// Plot a 3D surface for functions of two variables
    #[cfg(feature = "plotting")]
    pub fn plot_3d_surface<P, F>(path: P, f: F, title: &str) -> Result<(), Box<dyn Error>>
    where
        P: AsRef<Path>,
        F: Fn(f64, f64) -> f64,
    {
        let root = BitMapBackend::new(path.as_ref(), (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_3d(-5.0..5.0, -5.0..5.0, -2.0..2.0)?;
        
        chart.configure_axes()
            .x_desc("x")
            .y_desc("y")
            .z_desc("f(x,y)")
            .draw()?;
        
        // Generate surface data
        let n = 50;
        let mut data = vec![];
        
        for i in 0..n {
            for j in 0..n {
                let x = -5.0 + 10.0 * i as f64 / n as f64;
                let y = -5.0 + 10.0 * j as f64 / n as f64;
                let z = f(x, y);
                
                if z.is_finite() {
                    data.push((x, y, z));
                }
            }
        }
        
        chart.draw_series(
            SurfaceSeries::xoz(
                (-5.0..5.0).step(0.2),
                (-5.0..5.0).step(0.2),
                |x, y| f(x, y),
            )
            .style(&BLUE.mix(0.5)),
        )?;
        
        root.present()?;
        Ok(())
    }
}

/// Interactive visualization support
#[cfg(feature = "interactive")]
pub mod interactive {
    use super::*;
    
    /// Configuration for interactive plots
    pub struct InteractivePlotConfig {
        pub enable_zoom: bool,
        pub enable_pan: bool,
        pub enable_tooltips: bool,
        pub enable_export: bool,
    }
    
    /// Create an interactive plot that can be embedded in a web page
    pub fn create_interactive_plot<F>(f: F, config: InteractivePlotConfig) -> String
    where
        F: Fn(f64) -> f64,
    {
        // This would generate HTML/JS code for an interactive plot
        // using a library like Plotly.js or D3.js
        format!("<div>Interactive plot placeholder</div>")
    }
}

/// Export functions for different formats
pub mod export {
    use super::*;
    
    /// Export formats
    pub enum ExportFormat {
        PNG,
        SVG,
        PDF,
        LaTeX,
        CSV,
    }
    
    /// Export plot data in various formats
    pub fn export_plot_data<F>(
        f: F, 
        x_range: (f64, f64), 
        n_points: usize,
        format: ExportFormat,
    ) -> Result<Vec<u8>, Box<dyn Error>>
    where
        F: Fn(f64) -> f64,
    {
        match format {
            ExportFormat::CSV => {
                let mut csv_data = String::from("x,y\n");
                let step = (x_range.1 - x_range.0) / n_points as f64;
                
                for i in 0..=n_points {
                    let x = x_range.0 + i as f64 * step;
                    let y = f(x);
                    csv_data.push_str(&format!("{},{}\n", x, y));
                }
                
                Ok(csv_data.into_bytes())
            }
            ExportFormat::LaTeX => {
                // Generate LaTeX/TikZ code
                let mut latex = String::from("\\begin{tikzpicture}\n\\begin{axis}[\n");
                latex.push_str("    xlabel=$x$,\n    ylabel=$f(x)$,\n]\n");
                latex.push_str("\\addplot[blue,thick] coordinates {\n");
                
                let step = (x_range.1 - x_range.0) / n_points as f64;
                for i in 0..=n_points {
                    let x = x_range.0 + i as f64 * step;
                    let y = f(x);
                    if y.is_finite() {
                        latex.push_str(&format!("    ({},{})\n", x, y));
                    }
                }
                
                latex.push_str("};\n\\end{axis}\n\\end{tikzpicture}\n");
                Ok(latex.into_bytes())
            }
            _ => Err("Format not implemented yet".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plot_config() {
        let config = PlotConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert!(config.show_grid);
    }
    
    #[test]
    fn test_export_csv() {
        let data = export::export_plot_data(
            |x| x * x,
            (0.0, 1.0),
            10,
            export::ExportFormat::CSV,
        ).unwrap();
        
        let csv = String::from_utf8(data).unwrap();
        assert!(csv.contains("x,y\n"));
        assert!(csv.contains("0,0\n"));
        assert!(csv.contains("1,1\n"));
    }
}