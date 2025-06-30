//! CUDA kernels for GPU-accelerated optimization algorithms
//!
//! This module provides low-level CUDA kernel implementations for common
//! optimization operations, leveraging scirs2-core's GPU abstractions.

use crate::error::{ScirsError, ScirsResult};
use ndarray::{Array1, Array2};
use scirs2_core::gpu::{GpuDevice as GpuContext, GpuKernel};
use std::sync::Arc;

/// CUDA kernel for parallel function evaluation
pub struct FunctionEvaluationKernel {
    context: Arc<GpuContext>,
    kernel: GpuKernel,
}

impl FunctionEvaluationKernel {
    /// Create a new function evaluation kernel
    pub fn new(context: Arc<GpuContext>) -> ScirsResult<Self> {
        let kernel_source = r#"
            extern "C" __global__ void evaluate_batch(
                const double* points,
                double* results,
                int n_points,
                int n_dims,
                int function_type
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= n_points) return;
                
                const double* point = &points[idx * n_dims];
                double result = 0.0;
                
                // Switch based on function type
                switch (function_type) {
                    case 0: // Sphere function
                        for (int i = 0; i < n_dims; i++) {
                            result += point[i] * point[i];
                        }
                        break;
                        
                    case 1: // Rosenbrock function (2D only)
                        if (n_dims == 2) {
                            double x = point[0];
                            double y = point[1];
                            result = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
                        }
                        break;
                        
                    case 2: // Rastrigin function
                        result = 10.0 * n_dims;
                        for (int i = 0; i < n_dims; i++) {
                            result += point[i] * point[i] - 10.0 * cos(2.0 * M_PI * point[i]);
                        }
                        break;
                        
                    case 3: // Ackley function
                        {
                            double sum1 = 0.0, sum2 = 0.0;
                            for (int i = 0; i < n_dims; i++) {
                                sum1 += point[i] * point[i];
                                sum2 += cos(2.0 * M_PI * point[i]);
                            }
                            result = -20.0 * exp(-0.2 * sqrt(sum1 / n_dims)) - 
                                    exp(sum2 / n_dims) + 20.0 + M_E;
                        }
                        break;
                        
                    default:
                        // Default to sphere function
                        for (int i = 0; i < n_dims; i++) {
                            result += point[i] * point[i];
                        }
                        break;
                }
                
                results[idx] = result;
            }
        "#;

        let kernel = context.compile_kernel("evaluate_batch", kernel_source)?;

        Ok(Self { context, kernel })
    }

    /// Evaluate a batch of points using the specified function type
    pub fn evaluate_batch(
        &self,
        points: &GpuArray<f64>,
        function_type: i32,
        stream: Option<&GpuStream>,
    ) -> ScirsResult<GpuArray<f64>> {
        let (n_points, n_dims) = points.shape();
        let results = self.context.allocate_array::<f64>(&[n_points, 1])?;

        let block_size = 256;
        let grid_size = (n_points + block_size - 1) / block_size;

        self.kernel.launch(
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            &[
                points.as_ptr(),
                results.as_ptr(),
                &(n_points as i32),
                &(n_dims as i32),
                &function_type,
            ],
            stream,
        )?;

        Ok(results)
    }
}

/// CUDA kernel for gradient computation using finite differences
pub struct GradientKernel {
    context: Arc<GpuContext>,
    kernel: GpuKernel,
}

impl GradientKernel {
    /// Create a new gradient computation kernel
    pub fn new(context: Arc<GpuContext>) -> ScirsResult<Self> {
        let kernel_source = r#"
            extern "C" __global__ void compute_gradient_finite_diff(
                const double* points,
                const double* function_values,
                double* gradients,
                int n_points,
                int n_dims,
                double h,
                int function_type
            ) {
                int point_idx = blockIdx.x;
                int dim_idx = threadIdx.x;
                
                if (point_idx >= n_points || dim_idx >= n_dims) return;
                
                const double* point = &points[point_idx * n_dims];
                double* grad = &gradients[point_idx * n_dims];
                
                // Create perturbed point
                double perturbed_point[256]; // Assuming max 256 dimensions
                for (int i = 0; i < n_dims; i++) {
                    perturbed_point[i] = point[i];
                }
                
                // Forward difference
                perturbed_point[dim_idx] += h;
                double f_plus = evaluate_function(perturbed_point, n_dims, function_type);
                
                // Backward difference
                perturbed_point[dim_idx] = point[dim_idx] - h;
                double f_minus = evaluate_function(perturbed_point, n_dims, function_type);
                
                // Central difference
                grad[dim_idx] = (f_plus - f_minus) / (2.0 * h);
            }
            
            __device__ double evaluate_function(const double* point, int n_dims, int function_type) {
                double result = 0.0;
                
                switch (function_type) {
                    case 0: // Sphere function
                        for (int i = 0; i < n_dims; i++) {
                            result += point[i] * point[i];
                        }
                        break;
                        
                    case 1: // Rosenbrock function (2D only)
                        if (n_dims == 2) {
                            double x = point[0];
                            double y = point[1];
                            result = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
                        }
                        break;
                        
                    case 2: // Rastrigin function
                        result = 10.0 * n_dims;
                        for (int i = 0; i < n_dims; i++) {
                            result += point[i] * point[i] - 10.0 * cos(2.0 * M_PI * point[i]);
                        }
                        break;
                        
                    case 3: // Ackley function
                        {
                            double sum1 = 0.0, sum2 = 0.0;
                            for (int i = 0; i < n_dims; i++) {
                                sum1 += point[i] * point[i];
                                sum2 += cos(2.0 * M_PI * point[i]);
                            }
                            result = -20.0 * exp(-0.2 * sqrt(sum1 / n_dims)) - 
                                    exp(sum2 / n_dims) + 20.0 + M_E;
                        }
                        break;
                        
                    default:
                        for (int i = 0; i < n_dims; i++) {
                            result += point[i] * point[i];
                        }
                        break;
                }
                
                return result;
            }
        "#;

        let kernel = context.compile_kernel("compute_gradient_finite_diff", kernel_source)?;

        Ok(Self { context, kernel })
    }

    /// Compute gradients for a batch of points using finite differences
    pub fn compute_gradients(
        &self,
        points: &GpuArray<f64>,
        function_values: &GpuArray<f64>,
        function_type: i32,
        h: f64,
        stream: Option<&GpuStream>,
    ) -> ScirsResult<GpuArray<f64>> {
        let (n_points, n_dims) = points.shape();
        let gradients = self.context.allocate_array::<f64>(&[n_points, n_dims])?;

        self.kernel.launch(
            (n_points as u32, 1, 1),
            (n_dims as u32, 1, 1),
            &[
                points.as_ptr(),
                function_values.as_ptr(),
                gradients.as_ptr(),
                &(n_points as i32),
                &(n_dims as i32),
                &h,
                &function_type,
            ],
            stream,
        )?;

        Ok(gradients)
    }
}

/// CUDA kernel for differential evolution operations
pub struct DifferentialEvolutionKernel {
    context: Arc<GpuContext>,
    mutation_kernel: GpuKernel,
    crossover_kernel: GpuKernel,
    selection_kernel: GpuKernel,
}

impl DifferentialEvolutionKernel {
    /// Create a new differential evolution kernel
    pub fn new(context: Arc<GpuContext>) -> ScirsResult<Self> {
        let mutation_source = r#"
            extern "C" __global__ void differential_mutation(
                const double* population,
                double* trial_population,
                const int* random_indices,
                int pop_size,
                int n_dims,
                double f_scale
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= pop_size) return;
                
                int a = random_indices[idx * 3];
                int b = random_indices[idx * 3 + 1];
                int c = random_indices[idx * 3 + 2];
                
                for (int j = 0; j < n_dims; j++) {
                    trial_population[idx * n_dims + j] = 
                        population[a * n_dims + j] + 
                        f_scale * (population[b * n_dims + j] - population[c * n_dims + j]);
                }
            }
        "#;

        let crossover_source = r#"
            extern "C" __global__ void binomial_crossover(
                const double* population,
                double* trial_population,
                const double* random_values,
                const int* j_rand,
                int pop_size,
                int n_dims,
                double cr
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= pop_size) return;
                
                for (int j = 0; j < n_dims; j++) {
                    if (random_values[idx * n_dims + j] > cr && j != j_rand[idx]) {
                        trial_population[idx * n_dims + j] = population[idx * n_dims + j];
                    }
                }
            }
        "#;

        let selection_source = r#"
            extern "C" __global__ void selection(
                double* population,
                const double* trial_population,
                double* fitness,
                const double* trial_fitness,
                int pop_size,
                int n_dims
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= pop_size) return;
                
                if (trial_fitness[idx] <= fitness[idx]) {
                    fitness[idx] = trial_fitness[idx];
                    for (int j = 0; j < n_dims; j++) {
                        population[idx * n_dims + j] = trial_population[idx * n_dims + j];
                    }
                }
            }
        "#;

        let mutation_kernel = context.compile_kernel("differential_mutation", mutation_source)?;
        let crossover_kernel = context.compile_kernel("binomial_crossover", crossover_source)?;
        let selection_kernel = context.compile_kernel("selection", selection_source)?;

        Ok(Self {
            context,
            mutation_kernel,
            crossover_kernel,
            selection_kernel,
        })
    }

    /// Perform differential mutation on GPU
    pub fn mutation(
        &self,
        population: &GpuArray<f64>,
        trial_population: &GpuArray<f64>,
        random_indices: &GpuArray<i32>,
        f_scale: f64,
        stream: Option<&GpuStream>,
    ) -> ScirsResult<()> {
        let (pop_size, n_dims) = population.shape();
        let block_size = 256;
        let grid_size = (pop_size + block_size - 1) / block_size;

        self.mutation_kernel.launch(
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            &[
                population.as_ptr(),
                trial_population.as_ptr(),
                random_indices.as_ptr(),
                &(pop_size as i32),
                &(n_dims as i32),
                &f_scale,
            ],
            stream,
        )?;

        Ok(())
    }

    /// Perform binomial crossover on GPU
    pub fn crossover(
        &self,
        population: &GpuArray<f64>,
        trial_population: &GpuArray<f64>,
        random_values: &GpuArray<f64>,
        j_rand: &GpuArray<i32>,
        cr: f64,
        stream: Option<&GpuStream>,
    ) -> ScirsResult<()> {
        let (pop_size, n_dims) = population.shape();
        let block_size = 256;
        let grid_size = (pop_size + block_size - 1) / block_size;

        self.crossover_kernel.launch(
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            &[
                population.as_ptr(),
                trial_population.as_ptr(),
                random_values.as_ptr(),
                j_rand.as_ptr(),
                &(pop_size as i32),
                &(n_dims as i32),
                &cr,
            ],
            stream,
        )?;

        Ok(())
    }

    /// Perform selection on GPU
    pub fn selection(
        &self,
        population: &GpuArray<f64>,
        trial_population: &GpuArray<f64>,
        fitness: &GpuArray<f64>,
        trial_fitness: &GpuArray<f64>,
        stream: Option<&GpuStream>,
    ) -> ScirsResult<()> {
        let (pop_size, n_dims) = population.shape();
        let block_size = 256;
        let grid_size = (pop_size + block_size - 1) / block_size;

        self.selection_kernel.launch(
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            &[
                population.as_ptr(),
                trial_population.as_ptr(),
                fitness.as_ptr(),
                trial_fitness.as_ptr(),
                &(pop_size as i32),
                &(n_dims as i32),
            ],
            stream,
        )?;

        Ok(())
    }
}

/// CUDA kernel for particle swarm optimization operations
pub struct ParticleSwarmKernel {
    context: Arc<GpuContext>,
    update_kernel: GpuKernel,
}

impl ParticleSwarmKernel {
    /// Create a new particle swarm optimization kernel
    pub fn new(context: Arc<GpuContext>) -> ScirsResult<Self> {
        let kernel_source = r#"
            extern "C" __global__ void update_particles(
                double* positions,
                double* velocities,
                const double* personal_best,
                const double* global_best,
                const double* random_values,
                const double* bounds_low,
                const double* bounds_high,
                int swarm_size,
                int n_dims,
                double w,
                double c1,
                double c2
            ) {
                int particle_idx = blockIdx.x;
                int dim_idx = threadIdx.x;
                
                if (particle_idx >= swarm_size || dim_idx >= n_dims) return;
                
                int idx = particle_idx * n_dims + dim_idx;
                
                double r1 = random_values[idx * 2];
                double r2 = random_values[idx * 2 + 1];
                
                // Update velocity
                velocities[idx] = w * velocities[idx] + 
                    c1 * r1 * (personal_best[idx] - positions[idx]) +
                    c2 * r2 * (global_best[dim_idx] - positions[idx]);
                
                // Update position
                positions[idx] += velocities[idx];
                
                // Apply bounds
                if (positions[idx] < bounds_low[dim_idx]) {
                    positions[idx] = bounds_low[dim_idx];
                    velocities[idx] = 0.0;
                } else if (positions[idx] > bounds_high[dim_idx]) {
                    positions[idx] = bounds_high[dim_idx];
                    velocities[idx] = 0.0;
                }
            }
        "#;

        let kernel = context.compile_kernel("update_particles", kernel_source)?;

        Ok(Self { context, kernel })
    }

    /// Update particle positions and velocities on GPU
    pub fn update_particles(
        &self,
        positions: &GpuArray<f64>,
        velocities: &GpuArray<f64>,
        personal_best: &GpuArray<f64>,
        global_best: &GpuArray<f64>,
        random_values: &GpuArray<f64>,
        bounds_low: &GpuArray<f64>,
        bounds_high: &GpuArray<f64>,
        w: f64,
        c1: f64,
        c2: f64,
        stream: Option<&GpuStream>,
    ) -> ScirsResult<()> {
        let (swarm_size, n_dims) = positions.shape();

        self.kernel.launch(
            (swarm_size as u32, 1, 1),
            (n_dims as u32, 1, 1),
            &[
                positions.as_ptr(),
                velocities.as_ptr(),
                personal_best.as_ptr(),
                global_best.as_ptr(),
                random_values.as_ptr(),
                bounds_low.as_ptr(),
                bounds_high.as_ptr(),
                &(swarm_size as i32),
                &(n_dims as i32),
                &w,
                &c1,
                &c2,
            ],
            stream,
        )?;

        Ok(())
    }
}

/// CUDA kernel for reduction operations
pub struct ReductionKernel {
    context: Arc<GpuContext>,
    min_kernel: GpuKernel,
    max_kernel: GpuKernel,
    sum_kernel: GpuKernel,
}

impl ReductionKernel {
    /// Create a new reduction kernel
    pub fn new(context: Arc<GpuContext>) -> ScirsResult<Self> {
        let min_source = r#"
            extern "C" __global__ void reduce_min(
                const double* input,
                double* output,
                int* indices,
                int n
            ) {
                extern __shared__ double sdata[];
                extern __shared__ int sindices[];
                
                unsigned int tid = threadIdx.x;
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (i < n) {
                    sdata[tid] = input[i];
                    sindices[tid] = i;
                } else {
                    sdata[tid] = 1e308; // Large value
                    sindices[tid] = -1;
                }
                
                __syncthreads();
                
                for (unsigned int s = 1; s < blockDim.x; s *= 2) {
                    if (tid % (2 * s) == 0) {
                        if (sdata[tid + s] < sdata[tid]) {
                            sdata[tid] = sdata[tid + s];
                            sindices[tid] = sindices[tid + s];
                        }
                    }
                    __syncthreads();
                }
                
                if (tid == 0) {
                    output[blockIdx.x] = sdata[0];
                    indices[blockIdx.x] = sindices[0];
                }
            }
        "#;

        let max_source = r#"
            extern "C" __global__ void reduce_max(
                const double* input,
                double* output,
                int* indices,
                int n
            ) {
                extern __shared__ double sdata[];
                extern __shared__ int sindices[];
                
                unsigned int tid = threadIdx.x;
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (i < n) {
                    sdata[tid] = input[i];
                    sindices[tid] = i;
                } else {
                    sdata[tid] = -1e308; // Small value
                    sindices[tid] = -1;
                }
                
                __syncthreads();
                
                for (unsigned int s = 1; s < blockDim.x; s *= 2) {
                    if (tid % (2 * s) == 0) {
                        if (sdata[tid + s] > sdata[tid]) {
                            sdata[tid] = sdata[tid + s];
                            sindices[tid] = sindices[tid + s];
                        }
                    }
                    __syncthreads();
                }
                
                if (tid == 0) {
                    output[blockIdx.x] = sdata[0];
                    indices[blockIdx.x] = sindices[0];
                }
            }
        "#;

        let sum_source = r#"
            extern "C" __global__ void reduce_sum(
                const double* input,
                double* output,
                int n
            ) {
                extern __shared__ double sdata[];
                
                unsigned int tid = threadIdx.x;
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                
                sdata[tid] = (i < n) ? input[i] : 0.0;
                __syncthreads();
                
                for (unsigned int s = 1; s < blockDim.x; s *= 2) {
                    if (tid % (2 * s) == 0) {
                        sdata[tid] += sdata[tid + s];
                    }
                    __syncthreads();
                }
                
                if (tid == 0) {
                    output[blockIdx.x] = sdata[0];
                }
            }
        "#;

        let min_kernel = context.compile_kernel("reduce_min", min_source)?;
        let max_kernel = context.compile_kernel("reduce_max", max_source)?;
        let sum_kernel = context.compile_kernel("reduce_sum", sum_source)?;

        Ok(Self {
            context,
            min_kernel,
            max_kernel,
            sum_kernel,
        })
    }

    /// Find minimum value and its index
    pub fn reduce_min(&self, input: &GpuArray<f64>) -> ScirsResult<(f64, usize)> {
        let n = input.len();
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        let output = self.context.allocate_array::<f64>(&[grid_size])?;
        let indices = self.context.allocate_array::<i32>(&[grid_size])?;

        self.min_kernel.launch(
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            &[
                input.as_ptr(),
                output.as_ptr(),
                indices.as_ptr(),
                &(n as i32),
            ],
            None,
        )?;

        // Download results and find global minimum
        let output_cpu = self.context.download_array(&output)?;
        let indices_cpu = self.context.download_array(&indices)?;

        let mut min_val = output_cpu[[0, 0]];
        let mut min_idx = indices_cpu[[0, 0]] as usize;

        for i in 1..grid_size {
            if output_cpu[[i, 0]] < min_val {
                min_val = output_cpu[[i, 0]];
                min_idx = indices_cpu[[i, 0]] as usize;
            }
        }

        Ok((min_val, min_idx))
    }

    /// Find maximum value and its index
    pub fn reduce_max(&self, input: &GpuArray<f64>) -> ScirsResult<(f64, usize)> {
        let n = input.len();
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        let output = self.context.allocate_array::<f64>(&[grid_size])?;
        let indices = self.context.allocate_array::<i32>(&[grid_size])?;

        self.max_kernel.launch(
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            &[
                input.as_ptr(),
                output.as_ptr(),
                indices.as_ptr(),
                &(n as i32),
            ],
            None,
        )?;

        // Download results and find global maximum
        let output_cpu = self.context.download_array(&output)?;
        let indices_cpu = self.context.download_array(&indices)?;

        let mut max_val = output_cpu[[0, 0]];
        let mut max_idx = indices_cpu[[0, 0]] as usize;

        for i in 1..grid_size {
            if output_cpu[[i, 0]] > max_val {
                max_val = output_cpu[[i, 0]];
                max_idx = indices_cpu[[i, 0]] as usize;
            }
        }

        Ok((max_val, max_idx))
    }

    /// Sum all elements
    pub fn reduce_sum(&self, input: &GpuArray<f64>) -> ScirsResult<f64> {
        let n = input.len();
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        let output = self.context.allocate_array::<f64>(&[grid_size])?;

        self.sum_kernel.launch(
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            &[input.as_ptr(), output.as_ptr(), &(n as i32)],
            None,
        )?;

        // Download results and sum
        let output_cpu = self.context.download_array(&output)?;
        let sum = output_cpu.iter().sum::<f64>();

        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a CUDA-capable GPU and proper GPU context
    // They are primarily for documentation and will be skipped in CI

    #[test]
    #[ignore = "Requires CUDA GPU"]
    fn test_function_evaluation_kernel() {
        // This would test the function evaluation kernel with a mock GPU context
        // Implementation depends on the actual scirs2-core GPU infrastructure
    }

    #[test]
    #[ignore = "Requires CUDA GPU"]
    fn test_gradient_kernel() {
        // This would test the gradient computation kernel
    }

    #[test]
    #[ignore = "Requires CUDA GPU"]
    fn test_differential_evolution_kernel() {
        // This would test the differential evolution kernels
    }
}
