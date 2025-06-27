//! Search algorithms for Neural Architecture Search

use crate::error::Result;
use crate::nas::architecture_encoding::ArchitectureEncoding;
use crate::nas::SearchResult;
use std::sync::Arc;

/// Trait for search algorithms
pub trait SearchAlgorithm: Send + Sync {
    /// Propose architectures to evaluate
    fn propose_architectures(
        &self,
        history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>>;

    /// Update the algorithm with new results
    fn update(&mut self, results: &[SearchResult]) -> Result<()>;

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Random search algorithm
pub struct RandomSearch {
    seed: Option<u64>,
}

impl RandomSearch {
    /// Create a new random search algorithm
    pub fn new() -> Self {
        Self { seed: None }
    }

    /// Create with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self { seed: Some(seed) }
    }
}

impl SearchAlgorithm for RandomSearch {
    fn propose_architectures(
        &self,
        _history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        use rand::prelude::*;
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut rand::thread_rng()).unwrap()
        };

        let mut proposals = Vec::with_capacity(n_proposals);
        for _ in 0..n_proposals {
            // Create random architecture encoding
            let encoding = crate::nas::architecture_encoding::GraphEncoding::random(&mut rng)?;
            proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
        }

        Ok(proposals)
    }

    fn update(&mut self, _results: &[SearchResult]) -> Result<()> {
        // Random search doesn't learn from history
        Ok(())
    }

    fn name(&self) -> &str {
        "RandomSearch"
    }
}

/// Evolutionary search algorithm
pub struct EvolutionarySearch {
    population_size: usize,
    mutation_rate: f32,
    crossover_rate: f32,
    tournament_size: usize,
    elite_size: usize,
    population: Vec<Arc<dyn ArchitectureEncoding>>,
    fitness_scores: Vec<f64>,
}

impl EvolutionarySearch {
    /// Create a new evolutionary search algorithm
    pub fn new(population_size: usize) -> Self {
        Self {
            population_size,
            mutation_rate: 0.1,
            crossover_rate: 0.9,
            tournament_size: 3,
            elite_size: population_size / 10,
            population: Vec::new(),
            fitness_scores: Vec::new(),
        }
    }

    /// Set mutation rate
    pub fn with_mutation_rate(mut self, rate: f32) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// Set crossover rate
    pub fn with_crossover_rate(mut self, rate: f32) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// Tournament selection
    fn tournament_select(&self, rng: &mut impl rand::Rng) -> usize {
        use rand::prelude::*;
        
        let mut best_idx = rng.gen_range(0..self.population.len());
        let mut best_fitness = self.fitness_scores[best_idx];

        for _ in 1..self.tournament_size {
            let idx = rng.gen_range(0..self.population.len());
            if self.fitness_scores[idx] > best_fitness {
                best_idx = idx;
                best_fitness = self.fitness_scores[idx];
            }
        }

        best_idx
    }
}

impl SearchAlgorithm for EvolutionarySearch {
    fn propose_architectures(
        &self,
        history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        // Initialize population if empty
        if self.population.is_empty() {
            let mut proposals = Vec::with_capacity(n_proposals);
            for _ in 0..n_proposals {
                let encoding = crate::nas::architecture_encoding::GraphEncoding::random(&mut rng)?;
                proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
            }
            return Ok(proposals);
        }

        let mut proposals = Vec::with_capacity(n_proposals);

        // Elite selection
        let mut elite_indices: Vec<usize> = (0..self.population.len()).collect();
        elite_indices.sort_by(|&a, &b| {
            self.fitness_scores[b].partial_cmp(&self.fitness_scores[a]).unwrap()
        });
        
        for &idx in elite_indices.iter().take(self.elite_size.min(n_proposals)) {
            proposals.push(self.population[idx].clone());
        }

        // Generate offspring
        while proposals.len() < n_proposals {
            if rng.gen::<f32>() < self.crossover_rate && self.population.len() >= 2 {
                // Crossover
                let parent1_idx = self.tournament_select(&mut rng);
                let parent2_idx = self.tournament_select(&mut rng);
                
                if parent1_idx != parent2_idx {
                    let offspring = self.population[parent1_idx].crossover(
                        self.population[parent2_idx].as_ref()
                    )?;
                    proposals.push(Arc::from(offspring));
                } else {
                    // Fallback to mutation
                    let parent_idx = self.tournament_select(&mut rng);
                    let offspring = self.population[parent_idx].mutate(self.mutation_rate)?;
                    proposals.push(Arc::from(offspring));
                }
            } else {
                // Mutation
                let parent_idx = self.tournament_select(&mut rng);
                let offspring = self.population[parent_idx].mutate(self.mutation_rate)?;
                proposals.push(offspring);
            }
        }

        Ok(proposals)
    }

    fn update(&mut self, results: &[SearchResult]) -> Result<()> {
        // Update population with new results
        for result in results {
            self.population.push(result.architecture.clone());
            let fitness = result.metrics.values().sum::<f64>() / result.metrics.len() as f64;
            self.fitness_scores.push(fitness);
        }

        // Trim population to size
        if self.population.len() > self.population_size {
            let mut indices: Vec<usize> = (0..self.population.len()).collect();
            indices.sort_by(|&a, &b| {
                self.fitness_scores[b].partial_cmp(&self.fitness_scores[a]).unwrap()
            });

            let new_population: Vec<_> = indices.iter()
                .take(self.population_size)
                .map(|&idx| self.population[idx].clone())
                .collect();
            
            let new_scores: Vec<_> = indices.iter()
                .take(self.population_size)
                .map(|&idx| self.fitness_scores[idx])
                .collect();

            self.population = new_population;
            self.fitness_scores = new_scores;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "EvolutionarySearch"
    }
}

/// Reinforcement learning based search
pub struct ReinforcementSearch {
    controller_hidden_size: usize,
    learning_rate: f32,
    entropy_weight: f32,
    baseline_decay: f32,
    baseline: Option<f64>,
}

impl ReinforcementSearch {
    /// Create a new reinforcement learning search
    pub fn new() -> Self {
        Self {
            controller_hidden_size: 100,
            learning_rate: 3.5e-4,
            entropy_weight: 0.01,
            baseline_decay: 0.99,
            baseline: None,
        }
    }
}

impl SearchAlgorithm for ReinforcementSearch {
    fn propose_architectures(
        &self,
        _history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        // Simplified implementation - would use an RNN controller in practice
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let mut proposals = Vec::with_capacity(n_proposals);
        for _ in 0..n_proposals {
            let encoding = crate::nas::architecture_encoding::GraphEncoding::random(&mut rng)?;
            proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
        }

        Ok(proposals)
    }

    fn update(&mut self, results: &[SearchResult]) -> Result<()> {
        // Update baseline with exponential moving average
        let rewards: Vec<f64> = results.iter()
            .map(|r| r.metrics.values().sum::<f64>() / r.metrics.len() as f64)
            .collect();

        let mean_reward = if rewards.is_empty() {
            None
        } else {
            Some(rewards.iter().copied().sum::<f64>() / rewards.len() as f64)
        };
        
        if let Some(mean_reward) = mean_reward {
            self.baseline = Some(match self.baseline {
                Some(b) => self.baseline_decay as f64 * b + (1.0 - self.baseline_decay as f64) * mean_reward,
                None => mean_reward,
            });
        }

        // In practice, would update the controller network here
        Ok(())
    }

    fn name(&self) -> &str {
        "ReinforcementSearch"
    }
}

/// Differentiable architecture search (DARTS)
pub struct DifferentiableSearch {
    temperature: f64,
    arch_learning_rate: f32,
    weight_learning_rate: f32,
    arch_weight_decay: f32,
}

impl DifferentiableSearch {
    /// Create a new differentiable search
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            arch_learning_rate: 3e-4,
            weight_learning_rate: 0.025,
            arch_weight_decay: 1e-3,
        }
    }
}

impl SearchAlgorithm for DifferentiableSearch {
    fn propose_architectures(
        &self,
        _history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        // Simplified implementation - would use continuous relaxation in practice
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let mut proposals = Vec::with_capacity(n_proposals);
        for _ in 0..n_proposals {
            let encoding = crate::nas::architecture_encoding::SequentialEncoding::random(&mut rng)?;
            proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
        }

        Ok(proposals)
    }

    fn update(&mut self, _results: &[SearchResult]) -> Result<()> {
        // In practice, would update architecture parameters here
        Ok(())
    }

    fn name(&self) -> &str {
        "DifferentiableSearch"
    }
}

/// Bayesian optimization for architecture search
pub struct BayesianOptimization {
    surrogate_type: String,
    acquisition_function: String,
    n_initial_points: usize,
    xi: f64, // Exploration parameter
}

impl BayesianOptimization {
    /// Create a new Bayesian optimization search
    pub fn new() -> Self {
        Self {
            surrogate_type: "gaussian_process".to_string(),
            acquisition_function: "expected_improvement".to_string(),
            n_initial_points: 10,
            xi: 0.01,
        }
    }

    /// Set acquisition function
    pub fn with_acquisition(mut self, acquisition: &str) -> Self {
        self.acquisition_function = acquisition.to_string();
        self
    }
}

impl SearchAlgorithm for BayesianOptimization {
    fn propose_architectures(
        &self,
        history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        // If not enough initial points, do random search
        if history.len() < self.n_initial_points {
            let mut proposals = Vec::with_capacity(n_proposals);
            for _ in 0..n_proposals {
                let encoding = crate::nas::architecture_encoding::SequentialEncoding::random(&mut rng)?;
                proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
            }
            return Ok(proposals);
        }

        // In practice, would fit surrogate model and optimize acquisition function
        // For now, return random architectures with bias towards good regions
        let mut proposals = Vec::with_capacity(n_proposals);
        
        // Find best architecture so far
        let best_result = history.iter()
            .max_by(|a, b| {
                let a_score = a.metrics.values().sum::<f64>() / a.metrics.len() as f64;
                let b_score = b.metrics.values().sum::<f64>() / b.metrics.len() as f64;
                a_score.partial_cmp(&b_score).unwrap()
            });

        if let Some(best) = best_result {
            // Generate proposals near the best architecture
            for _ in 0..n_proposals {
                let mutated = best.architecture.mutate(0.1)?;
                proposals.push(Arc::from(mutated));
            }
        } else {
            // Fallback to random
            for _ in 0..n_proposals {
                let encoding = crate::nas::architecture_encoding::SequentialEncoding::random(&mut rng)?;
                proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
            }
        }

        Ok(proposals)
    }

    fn update(&mut self, _results: &[SearchResult]) -> Result<()> {
        // In practice, would update the surrogate model here
        Ok(())
    }

    fn name(&self) -> &str {
        "BayesianOptimization"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::EvaluationMetrics;

    fn create_dummy_result() -> SearchResult {
        let encoding = crate::nas::architecture_encoding::SequentialEncoding::random(
            &mut rand::thread_rng()
        ).unwrap();
        
        let mut metrics = EvaluationMetrics::new();
        metrics.insert("accuracy".to_string(), 0.95);
        
        SearchResult {
            architecture: Arc::new(encoding),
            metrics,
            training_time: 100.0,
            parameter_count: 1000000,
            flops: Some(1000000),
        }
    }

    #[test]
    fn test_random_search() {
        let search = RandomSearch::new();
        let proposals = search.propose_architectures(&[], 5).unwrap();
        assert_eq!(proposals.len(), 5);
    }

    #[test]
    fn test_evolutionary_search() {
        let mut search = EvolutionarySearch::new(10);
        let proposals = search.propose_architectures(&[], 5).unwrap();
        assert_eq!(proposals.len(), 5);

        // Test update
        let results = vec![create_dummy_result(); 5];
        search.update(&results).unwrap();
    }

    #[test]
    fn test_reinforcement_search() {
        let mut search = ReinforcementSearch::new();
        let proposals = search.propose_architectures(&[], 3).unwrap();
        assert_eq!(proposals.len(), 3);

        // Test update
        let results = vec![create_dummy_result(); 3];
        search.update(&results).unwrap();
        assert!(search.baseline.is_some());
    }
}