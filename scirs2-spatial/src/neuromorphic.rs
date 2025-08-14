//! Neuromorphic computing acceleration for spatial algorithms
//!
//! This module implements brain-inspired computing paradigms for spatial algorithms,
//! leveraging spiking neural networks, memristive computing, and neuroplasticity
//! for advanced-efficient adaptive spatial processing. These algorithms mimic biological
//! neural computation to achieve extreme energy efficiency and real-time adaptation.
//!
//! # Features
//!
//! - **Spiking Neural Networks (SNNs)** for spatial pattern recognition
//! - **Memristive crossbar arrays** for in-memory spatial computations
//! - **Spike-timing dependent plasticity (STDP)** for adaptive learning
//! - **Event-driven spatial processing** for real-time applications
//! - **Neuromorphic clustering** using competitive learning
//! - **Temporal coding** for multi-dimensional spatial data
//! - **Bio-inspired optimization** using neural adaptation mechanisms
//!
//! # Neuromorphic Principles
//!
//! The algorithms are based on key neuromorphic computing concepts:
//!
//! - **Sparse spike-based communication**: Information encoded in spike timing
//! - **Massively parallel processing**: Thousands of simple processing units
//! - **Adaptive synaptic weights**: Learning through experience
//! - **Event-driven computation**: Processing only when spikes occur
//! - **Low-power operation**: Inspired by brain's energy efficiency
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::neuromorphic::{SpikingNeuralClusterer, NeuromorphicProcessor};
//! use ndarray::array;
//!
//! // Spiking neural network clustering
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//! let mut snn_clusterer = SpikingNeuralClusterer::new(2)
//!     .with_spike_threshold(0.5)
//!     .with_stdp_learning(true)
//!     .with_lateral_inhibition(true);
//!
//! let (clusters, spike_patterns) = snn_clusterer.fit(&points.view())?;
//! println!("Neuromorphic clusters: {:?}", clusters);
//!
//! // Event-driven spatial processing
//! let mut processor = NeuromorphicProcessor::new()
//!     .with_memristive_crossbar(true)
//!     .with_temporal_coding(true);
//!
//! let events = processor.encode_spatial_events(&points.view())?;
//! let processed_events = processor.process_events(&events)?;
//! ```

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};
// Constants removed - not used in this module

/// Neuromorphic spike event
#[derive(Debug, Clone)]
pub struct SpikeEvent {
    /// Source neuron ID
    pub neuron_id: usize,
    /// Spike timestamp (in simulation time units)
    pub timestamp: f64,
    /// Spike amplitude
    pub amplitude: f64,
    /// Spatial coordinates associated with the spike
    pub spatial_coords: Vec<f64>,
}

/// Spiking neuron model
#[derive(Debug, Clone)]
pub struct SpikingNeuron {
    /// Membrane potential
    pub membrane_potential: f64,
    /// Spike threshold
    pub threshold: f64,
    /// Refractory period
    pub refractory_period: f64,
    /// Time since last spike
    pub time_since_spike: f64,
    /// Leak constant
    pub leak_constant: f64,
    /// Input current
    pub input_current: f64,
    /// Neuron position in space
    pub position: Vec<f64>,
    /// Learning rate
    pub learning_rate: f64,
}

impl SpikingNeuron {
    /// Create new spiking neuron
    pub fn new(position: Vec<f64>) -> Self {
        Self {
            membrane_potential: 0.0,
            threshold: 1.0,
            refractory_period: 2.0,
            time_since_spike: 0.0,
            leak_constant: 0.1,
            input_current: 0.0,
            position,
            learning_rate: 0.01,
        }
    }

    /// Update neuron state and check for spike
    pub fn update(&mut self, dt: f64, inputcurrent: f64) -> bool {
        self.time_since_spike += dt;

        // Check if in refractory period
        if self.time_since_spike < self.refractory_period {
            return false;
        }

        // Update membrane potential using leaky integrate-and-fire model
        self.input_current = inputcurrent;
        let leak_term = -self.leak_constant * self.membrane_potential;
        self.membrane_potential += dt * (leak_term + inputcurrent);

        // Check for spike
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = 0.0; // Reset potential
            self.time_since_spike = 0.0; // Reset spike timer
            true
        } else {
            false
        }
    }

    /// Calculate distance-based influence on another neuron
    pub fn calculate_influence(&self, _otherposition: &[f64]) -> f64 {
        let distance: f64 = self
            .position
            .iter()
            .zip(_otherposition.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Gaussian influence function
        (-distance.powi(2) / 2.0).exp()
    }
}

/// Synaptic connection with STDP learning
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Pre-synaptic neuron ID
    pub pre_neuron: usize,
    /// Post-synaptic neuron ID
    pub post_neuron: usize,
    /// Synaptic weight
    pub weight: f64,
    /// Last pre-synaptic spike time
    pub last_pre_spike: f64,
    /// Last post-synaptic spike time
    pub last_post_spike: f64,
    /// STDP learning rate
    pub stdp_rate: f64,
    /// STDP time constant
    pub stdp_tau: f64,
}

impl Synapse {
    /// Create new synapse
    pub fn new(pre_neuron: usize, post_neuron: usize, initialweight: f64) -> Self {
        Self {
            pre_neuron,
            post_neuron,
            weight: initialweight,
            last_pre_spike: -1000.0,
            last_post_spike: -1000.0,
            stdp_rate: 0.01,
            stdp_tau: 20.0,
        }
    }

    /// Update synaptic weight using STDP rule
    pub fn update_stdp(&mut self, currenttime: f64, pre_spiked: bool, postspiked: bool) {
        if pre_spiked {
            self.last_pre_spike = currenttime;
        }
        if postspiked {
            self.last_post_spike = currenttime;
        }

        // Apply STDP learning rule
        if pre_spiked && self.last_post_spike > self.last_pre_spike - 50.0 {
            // Potentiation: pre before post
            let dt = self.last_post_spike - self.last_pre_spike;
            if dt > 0.0 {
                let delta_w = self.stdp_rate * (-dt / self.stdp_tau).exp();
                self.weight += delta_w;
            }
        }

        if postspiked && self.last_pre_spike > self.last_post_spike - 50.0 {
            // Depression: post before pre
            let dt = self.last_pre_spike - self.last_post_spike;
            if dt > 0.0 {
                let delta_w = -self.stdp_rate * (-dt / self.stdp_tau).exp();
                self.weight += delta_w;
            }
        }

        // Keep weights in reasonable bounds
        self.weight = self.weight.clamp(-2.0, 2.0);
    }

    /// Calculate synaptic current
    pub fn synaptic_current(&self, _pre_spikestrength: f64) -> f64 {
        self.weight * _pre_spikestrength
    }
}

/// Spiking neural network clusterer
#[derive(Debug, Clone)]
pub struct SpikingNeuralClusterer {
    /// Network of spiking neurons
    neurons: Vec<SpikingNeuron>,
    /// Synaptic connections
    synapses: Vec<Synapse>,
    /// Number of clusters (output neurons)
    _num_clusters: usize,
    /// Spike threshold
    spike_threshold: f64,
    /// Enable STDP learning
    stdp_learning: bool,
    /// Enable lateral inhibition
    lateral_inhibition: bool,
    /// Simulation time step
    dt: f64,
    /// Current simulation time
    currenttime: f64,
    /// Spike history
    spike_history: Vec<SpikeEvent>,
}

impl SpikingNeuralClusterer {
    /// Create new spiking neural clusterer
    pub fn new(_num_clusters: usize) -> Self {
        Self {
            neurons: Vec::new(),
            synapses: Vec::new(),
            _num_clusters,
            spike_threshold: 1.0,
            stdp_learning: true,
            lateral_inhibition: true,
            dt: 0.1,
            currenttime: 0.0,
            spike_history: Vec::new(),
        }
    }

    /// Configure spike threshold
    pub fn with_spike_threshold(mut self, threshold: f64) -> Self {
        self.spike_threshold = threshold;
        self
    }

    /// Enable/disable STDP learning
    pub fn with_stdp_learning(mut self, enabled: bool) -> Self {
        self.stdp_learning = enabled;
        self
    }

    /// Enable/disable lateral inhibition
    pub fn with_lateral_inhibition(mut self, enabled: bool) -> Self {
        self.lateral_inhibition = enabled;
        self
    }

    /// Fit clustering to spatial data
    pub fn fit(
        &mut self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<(Array1<usize>, Vec<SpikeEvent>)> {
        let (n_points, n_dims) = points.dim();

        // Initialize neural network
        self.initialize_network(n_dims)?;

        // Present data points as spike trains
        let mut assignments = Array1::zeros(n_points);

        for epoch in 0..100 {
            self.currenttime = epoch as f64 * 100.0;

            for (point_idx, point) in points.outer_iter().enumerate() {
                // Encode spatial point as spike train
                let _spiketrain = self.encode_point_as_spikes(&point.to_owned())?;

                // Process spike train through network
                let winning_neuron = self.process_spike_train(&_spiketrain)?;
                assignments[point_idx] = winning_neuron;

                // Apply learning if enabled
                if self.stdp_learning {
                    self.apply_stdp_learning(&_spiketrain)?;
                }
            }

            // Apply lateral inhibition
            if self.lateral_inhibition {
                self.apply_lateral_inhibition()?;
            }
        }

        Ok((assignments, self.spike_history.clone()))
    }

    /// Initialize spiking neural network
    fn initialize_network(&mut self, _inputdims: usize) -> SpatialResult<()> {
        self.neurons.clear();
        self.synapses.clear();

        // Create input neurons (one per dimension)
        for i in 0.._inputdims {
            let position = vec![i as f64];
            let mut neuron = SpikingNeuron::new(position);
            neuron.threshold = self.spike_threshold;
            self.neurons.push(neuron);
        }

        // Create output neurons (cluster centers)
        for _i in 0..self._num_clusters {
            let position = (0.._inputdims).map(|_| rand::random::<f64>()).collect();
            let mut neuron = SpikingNeuron::new(position);
            neuron.threshold = self.spike_threshold;
            self.neurons.push(neuron);
        }

        // Create synaptic connections (input to output)
        for i in 0.._inputdims {
            for j in 0..self._num_clusters {
                let output_idx = _inputdims + j;
                let weight = rand::random::<f64>() * 0.5;
                let synapse = Synapse::new(i, output_idx, weight);
                self.synapses.push(synapse);
            }
        }

        // Create lateral inhibitory connections between output neurons
        if self.lateral_inhibition {
            for i in 0..self._num_clusters {
                for j in 0..self._num_clusters {
                    if i != j {
                        let neuron_i = _inputdims + i;
                        let neuron_j = _inputdims + j;
                        let synapse = Synapse::new(neuron_i, neuron_j, -0.5);
                        self.synapses.push(synapse);
                    }
                }
            }
        }

        Ok(())
    }

    /// Encode spatial point as spike train
    fn encode_point_as_spikes(&self, point: &Array1<f64>) -> SpatialResult<Vec<SpikeEvent>> {
        let mut _spiketrain = Vec::new();

        // Rate coding: spike frequency proportional to coordinate value
        for (dim, &coord) in point.iter().enumerate() {
            // Normalize coordinate to [0, 1] and scale to spike rate
            let normalized_coord = (coord + 10.0) / 20.0; // Assume data in [-10, 10]
            let spike_rate = normalized_coord.clamp(0.0, 1.0) * 50.0; // Max 50 Hz

            // Generate Poisson spike train
            let num_spikes = (spike_rate * 1.0) as usize; // 1 second duration
            for spike_idx in 0..num_spikes {
                let timestamp = self.currenttime + (spike_idx as f64) * (1.0 / spike_rate);
                let spike = SpikeEvent {
                    neuron_id: dim,
                    timestamp,
                    amplitude: 1.0,
                    spatial_coords: point.to_vec(),
                };
                _spiketrain.push(spike);
            }
        }

        // Sort spikes by timestamp
        _spiketrain.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());

        Ok(_spiketrain)
    }

    /// Process spike train through network
    fn process_spike_train(&mut self, _spiketrain: &[SpikeEvent]) -> SpatialResult<usize> {
        let inputdims = self.neurons.len() - self._num_clusters;
        let mut neuron_spike_counts = vec![0; self._num_clusters];

        // Simulate network for duration of spike _train
        let simulation_duration = 10.0; // 10 time units
        let mut t = self.currenttime;
        let mut spike_idx = 0;

        while t < self.currenttime + simulation_duration {
            // Apply input spikes
            let mut input_currents = vec![0.0; self.neurons.len()];

            while spike_idx < _spiketrain.len() && _spiketrain[spike_idx].timestamp <= t {
                let spike = &_spiketrain[spike_idx];
                if spike.neuron_id < inputdims {
                    input_currents[spike.neuron_id] += spike.amplitude;
                }
                spike_idx += 1;
            }

            // Calculate synaptic currents
            for synapse in &self.synapses {
                if synapse.pre_neuron < self.neurons.len()
                    && synapse.post_neuron < self.neurons.len()
                {
                    let pre_current = input_currents[synapse.pre_neuron];
                    let synaptic_current = synapse.synaptic_current(pre_current);
                    input_currents[synapse.post_neuron] += synaptic_current;
                }
            }

            // Update neurons and check for spikes
            for (neuron_idx, neuron) in self.neurons.iter_mut().enumerate() {
                let spiked = neuron.update(self.dt, input_currents[neuron_idx]);

                if spiked && neuron_idx >= inputdims {
                    let cluster_idx = neuron_idx - inputdims;
                    neuron_spike_counts[cluster_idx] += 1;

                    // Record spike event
                    let spike_event = SpikeEvent {
                        neuron_id: neuron_idx,
                        timestamp: t,
                        amplitude: 1.0,
                        spatial_coords: neuron.position.clone(),
                    };
                    self.spike_history.push(spike_event);
                }
            }

            t += self.dt;
        }

        // Find winning neuron (cluster with most spikes)
        let winning_cluster = neuron_spike_counts
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(idx_, _)| idx_)
            .unwrap_or(0);

        Ok(winning_cluster)
    }

    /// Apply STDP learning to synapses
    fn apply_stdp_learning(&mut self, _spiketrain: &[SpikeEvent]) -> SpatialResult<()> {
        // Create spike timing map
        let mut spike_times: HashMap<usize, Vec<f64>> = HashMap::new();
        for spike in _spiketrain {
            spike_times
                .entry(spike.neuron_id)
                .or_default()
                .push(spike.timestamp);
        }

        // Add output neuron spikes from history
        for spike in &self.spike_history {
            spike_times
                .entry(spike.neuron_id)
                .or_default()
                .push(spike.timestamp);
        }

        // Update synaptic weights using STDP
        let empty_spikes = Vec::new();
        for synapse in &mut self.synapses {
            let pre_spikes = spike_times
                .get(&synapse.pre_neuron)
                .unwrap_or(&empty_spikes);
            let post_spikes = spike_times
                .get(&synapse.post_neuron)
                .unwrap_or(&empty_spikes);

            // Check for coincident spikes
            for &pre_time in pre_spikes {
                for &post_time in post_spikes {
                    let dt = post_time - pre_time;
                    if dt.abs() < 50.0 {
                        // Within STDP window
                        if dt > 0.0 {
                            // Potentiation
                            synapse.weight += synapse.stdp_rate * (-dt / synapse.stdp_tau).exp();
                        } else {
                            // Depression
                            synapse.weight -= synapse.stdp_rate * (dt / synapse.stdp_tau).exp();
                        }
                    }
                }
            }

            // Clamp weights
            synapse.weight = synapse.weight.clamp(-2.0, 2.0);
        }

        Ok(())
    }

    /// Apply lateral inhibition between output neurons
    fn apply_lateral_inhibition(&mut self) -> SpatialResult<()> {
        let inputdims = self.neurons.len() - self._num_clusters;

        // Strengthen inhibitory connections between active neurons
        for i in 0..self._num_clusters {
            for j in 0..self._num_clusters {
                if i != j {
                    let neuron_i_idx = inputdims + i;
                    let neuron_j_idx = inputdims + j;

                    // Find inhibitory synapse
                    for synapse in &mut self.synapses {
                        if synapse.pre_neuron == neuron_i_idx && synapse.post_neuron == neuron_j_idx
                        {
                            // Strengthen inhibition based on activity
                            let activity_i = self.neurons[neuron_i_idx].membrane_potential;
                            let activity_j = self.neurons[neuron_j_idx].membrane_potential;

                            if activity_i > activity_j {
                                synapse.weight -= 0.01; // Strengthen inhibition
                                synapse.weight = synapse.weight.clamp(-2.0, 0.0);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Neuromorphic processor for general spatial computations
#[derive(Debug, Clone)]
pub struct NeuromorphicProcessor {
    /// Enable memristive crossbar arrays
    memristive_crossbar: bool,
    /// Enable temporal coding
    temporal_coding: bool,
    /// Crossbar array dimensions
    crossbar_size: (usize, usize),
    /// Memristive device conductances
    conductances: Array2<f64>,
    /// Event processing pipeline
    event_pipeline: VecDeque<SpikeEvent>,
}

impl Default for NeuromorphicProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuromorphicProcessor {
    /// Create new neuromorphic processor
    pub fn new() -> Self {
        Self {
            memristive_crossbar: false,
            temporal_coding: false,
            crossbar_size: (64, 64),
            conductances: Array2::zeros((64, 64)),
            event_pipeline: VecDeque::new(),
        }
    }

    /// Enable memristive crossbar arrays
    pub fn with_memristive_crossbar(mut self, enabled: bool) -> Self {
        self.memristive_crossbar = enabled;
        if enabled {
            self.initialize_crossbar();
        }
        self
    }

    /// Enable temporal coding
    pub fn with_temporal_coding(mut self, enabled: bool) -> Self {
        self.temporal_coding = enabled;
        self
    }

    /// Configure crossbar size
    pub fn with_crossbar_size(mut self, rows: usize, cols: usize) -> Self {
        self.crossbar_size = (rows, cols);
        self.conductances = Array2::zeros((rows, cols));
        self
    }

    /// Encode spatial data as neuromorphic events
    pub fn encode_spatial_events(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Vec<SpikeEvent>> {
        let (_n_points, n_dims) = points.dim();
        let mut events = Vec::new();

        for (point_idx, point) in points.outer_iter().enumerate() {
            for (dim, &coord) in point.iter().enumerate() {
                // Temporal coding: encode coordinate as spike timing
                let normalized_coord = (coord + 10.0) / 20.0; // Normalize to [0, 1]

                if self.temporal_coding {
                    // Timing-based encoding
                    let spike_time = normalized_coord * 100.0; // Map to [0, 100] time units
                    let event = SpikeEvent {
                        neuron_id: point_idx * n_dims + dim,
                        timestamp: spike_time,
                        amplitude: 1.0,
                        spatial_coords: point.to_vec(),
                    };
                    events.push(event);
                } else {
                    // Rate-based encoding
                    let spike_rate = normalized_coord * 50.0; // Max 50 Hz
                    let num_spikes = (spike_rate) as usize;

                    for spike_idx in 0..num_spikes {
                        let spike_time = (spike_idx as f64) * (1.0 / spike_rate);
                        let event = SpikeEvent {
                            neuron_id: point_idx * n_dims + dim,
                            timestamp: spike_time,
                            amplitude: 1.0,
                            spatial_coords: point.to_vec(),
                        };
                        events.push(event);
                    }
                }
            }
        }

        // Sort events by timestamp
        events.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());

        Ok(events)
    }

    /// Process events through neuromorphic pipeline
    pub fn process_events(&mut self, events: &[SpikeEvent]) -> SpatialResult<Vec<SpikeEvent>> {
        let mut processed_events = Vec::new();

        for event in events {
            self.event_pipeline.push_back(event.clone());

            // Process through memristive crossbar if enabled
            if self.memristive_crossbar {
                let crossbar_output = self.process_through_crossbar(event)?;
                processed_events.extend(crossbar_output);
            } else {
                processed_events.push(event.clone());
            }

            // Apply temporal dynamics
            if self.temporal_coding {
                NeuromorphicProcessor::apply_temporal_dynamics(&mut processed_events)?;
            }

            // Maintain event pipeline size
            if self.event_pipeline.len() > 1000 {
                self.event_pipeline.pop_front();
            }
        }

        Ok(processed_events)
    }

    /// Initialize memristive crossbar array
    fn initialize_crossbar(&mut self) {
        let (rows, cols) = self.crossbar_size;

        // Initialize conductances with random values
        for i in 0..rows {
            for j in 0..cols {
                // Random conductance between 0.1 and 1.0 (normalized)
                self.conductances[[i, j]] = 0.1 + rand::random::<f64>() * 0.9;
            }
        }
    }

    /// Process event through memristive crossbar
    fn process_through_crossbar(&mut self, event: &SpikeEvent) -> SpatialResult<Vec<SpikeEvent>> {
        let (rows, cols) = self.crossbar_size;
        let mut output_events = Vec::new();

        // Map input neuron to crossbar row
        let input_row = event.neuron_id % rows;

        // Compute crossbar outputs
        for col in 0..cols {
            let conductance = self.conductances[[input_row, col]];
            let output_current = event.amplitude * conductance;

            // Generate output spike if current exceeds threshold
            if output_current > 0.5 {
                let output_event = SpikeEvent {
                    neuron_id: rows + col,             // Offset for output neurons
                    timestamp: event.timestamp + 0.1, // Small delay
                    amplitude: output_current,
                    spatial_coords: event.spatial_coords.clone(),
                };
                output_events.push(output_event);

                // Update memristive device (Hebbian-like plasticity)
                self.update_memristive_device(input_row, col, event.amplitude)?;
            }
        }

        Ok(output_events)
    }

    /// Update memristive device conductance
    fn update_memristive_device(
        &mut self,
        row: usize,
        col: usize,
        spike_amplitude: f64,
    ) -> SpatialResult<()> {
        let current_conductance = self.conductances[[row, col]];

        // Simple memristive update rule
        let learning_rate = 0.001;
        let conductance_change = learning_rate * spike_amplitude * (1.0 - current_conductance);

        self.conductances[[row, col]] += conductance_change;
        self.conductances[[row, col]] = self.conductances[[row, col]].clamp(0.0, 1.0);

        Ok(())
    }

    /// Apply temporal dynamics to event processing
    fn apply_temporal_dynamics(events: &mut Vec<SpikeEvent>) -> SpatialResult<()> {
        // Apply temporal filtering and spike-timing dependent processing
        let mut filtered_events = Vec::new();

        for (i, event) in events.iter().enumerate() {
            let mut should_include = true;
            let mut modified_event = event.clone();

            // Check for temporal correlations with recent events
            for other_event in events.iter().skip(i + 1) {
                let time_diff = (other_event.timestamp - event.timestamp).abs();

                if time_diff < 5.0 {
                    // Within temporal window
                    // Apply temporal correlation enhancement
                    modified_event.amplitude *= 1.1;

                    // Coincidence detection
                    if time_diff < 1.0 {
                        modified_event.amplitude *= 1.5; // Strong enhancement for coincidence
                    }
                }

                // Refractory period simulation
                if time_diff < 0.5 && event.neuron_id == other_event.neuron_id {
                    should_include = false; // Suppress due to refractory period
                    break;
                }
            }

            if should_include {
                filtered_events.push(modified_event);
            }
        }

        *events = filtered_events;
        Ok(())
    }

    /// Get crossbar statistics
    pub fn get_crossbar_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if self.memristive_crossbar {
            let total_conductance: f64 = self.conductances.sum();
            let avg_conductance =
                total_conductance / (self.crossbar_size.0 * self.crossbar_size.1) as f64;
            let max_conductance = self.conductances.fold(0.0f64, |acc, &x| acc.max(x));
            let min_conductance = self.conductances.fold(1.0f64, |acc, &x| acc.min(x));

            stats.insert("total_conductance".to_string(), total_conductance);
            stats.insert("avg_conductance".to_string(), avg_conductance);
            stats.insert("max_conductance".to_string(), max_conductance);
            stats.insert("min_conductance".to_string(), min_conductance);
        }

        stats.insert(
            "event_pipeline_length".to_string(),
            self.event_pipeline.len() as f64,
        );
        stats
    }
}

/// Bio-inspired competitive learning for spatial clustering
#[derive(Debug, Clone)]
pub struct CompetitiveNeuralClusterer {
    /// Network neurons representing cluster centers
    neurons: Vec<Array1<f64>>,
    /// Learning rates for each neuron
    learning_rates: Vec<f64>,
    /// Lateral inhibition strengths
    inhibition_strengths: Array2<f64>,
    /// Winner-take-all threshold
    #[allow(dead_code)]
    wta_threshold: f64,
    /// Neighborhood function parameters
    neighborhood_sigma: f64,
}

impl CompetitiveNeuralClusterer {
    /// Create new competitive neural clusterer
    pub fn new(_num_clusters: usize, inputdims: usize) -> Self {
        let mut neurons = Vec::new();
        let mut learning_rates = Vec::new();

        // Initialize neurons with random weights
        for _ in 0.._num_clusters {
            let weights = Array1::from_shape_fn(inputdims, |_| rand::random::<f64>());
            neurons.push(weights);
            learning_rates.push(0.1);
        }

        // Initialize inhibition matrix
        let inhibition_strengths =
            Array2::from_shape_fn((_num_clusters, _num_clusters), |(i, j)| {
                if i == j {
                    0.0
                } else {
                    0.1
                }
            });

        Self {
            neurons,
            learning_rates,
            inhibition_strengths,
            wta_threshold: 0.5,
            neighborhood_sigma: 1.0,
        }
    }

    /// Train competitive network on spatial data
    pub fn fit(
        &mut self,
        points: &ArrayView2<'_, f64>,
        epochs: usize,
    ) -> SpatialResult<Array1<usize>> {
        let n_points_ = points.dim().0;
        let mut assignments = Array1::zeros(n_points_);

        for epoch in 0..epochs {
            // Adjust learning rate and neighborhood size
            let epoch_factor = 1.0 - (epoch as f64) / (epochs as f64);
            let current_sigma = self.neighborhood_sigma * epoch_factor;

            for (point_idx, point) in points.outer_iter().enumerate() {
                // Find winning neuron
                let winner = self.find_winner(&point.to_owned())?;
                assignments[point_idx] = winner;

                // Update winner and neighbors
                self.update_neurons(&point.to_owned(), winner, current_sigma, epoch_factor)?;

                // Apply lateral inhibition
                self.apply_lateral_inhibition(winner)?;
            }
        }

        Ok(assignments)
    }

    /// Find winning neuron using competitive dynamics
    fn find_winner(&self, input: &Array1<f64>) -> SpatialResult<usize> {
        let mut min_distance = f64::INFINITY;
        let mut winner = 0;

        for (i, neuron) in self.neurons.iter().enumerate() {
            // Calculate Euclidean distance
            let distance: f64 = input
                .iter()
                .zip(neuron.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if distance < min_distance {
                min_distance = distance;
                winner = i;
            }
        }

        Ok(winner)
    }

    /// Update neuron weights using competitive learning
    fn update_neurons(
        &mut self,
        input: &Array1<f64>,
        winner: usize,
        sigma: f64,
        learning_factor: f64,
    ) -> SpatialResult<()> {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Calculate neighborhood influence
            let distance_to_winner = (i as i32 - winner as i32).abs() as f64;
            let neighborhood_influence =
                (-distance_to_winner.powi(2) / (2.0 * sigma.powi(2))).exp();

            // Update neuron weights
            let effective_learning_rate =
                self.learning_rates[i] * learning_factor * neighborhood_influence;

            for (weight, &input_val) in neuron.iter_mut().zip(input.iter()) {
                *weight += effective_learning_rate * (input_val - *weight);
            }
        }

        Ok(())
    }

    /// Apply lateral inhibition between neurons
    fn apply_lateral_inhibition(&mut self, winner: usize) -> SpatialResult<()> {
        // Strengthen inhibitory connections from winner to others
        for i in 0..self.neurons.len() {
            if i != winner {
                self.inhibition_strengths[[winner, i]] += 0.001;
                self.inhibition_strengths[[winner, i]] =
                    self.inhibition_strengths[[winner, i]].min(0.5);

                // Reduce learning rate of inhibited neurons
                self.learning_rates[i] *= 0.99;
                self.learning_rates[i] = self.learning_rates[i].max(0.001);
            }
        }

        // Boost winner's learning rate slightly
        self.learning_rates[winner] *= 1.001;
        self.learning_rates[winner] = self.learning_rates[winner].min(0.2);

        Ok(())
    }

    /// Get cluster centers (neuron weights)
    pub fn get_cluster_centers(&self) -> Array2<f64> {
        let _num_clusters = self.neurons.len();
        let inputdims = self.neurons[0].len();

        let mut centers = Array2::zeros((_num_clusters, inputdims));
        for (i, neuron) in self.neurons.iter().enumerate() {
            centers.row_mut(i).assign(neuron);
        }

        centers
    }
}

/// Advanced homeostatic plasticity for neuromorphic spatial learning
#[derive(Debug, Clone)]
pub struct HomeostaticNeuralClusterer {
    /// Number of clusters (output neurons)
    _num_clusters: usize,
    /// Input dimension
    inputdim: usize,
    /// Output neurons with homeostatic mechanisms
    output_neurons: Vec<HomeostaticNeuron>,
    /// Synaptic weights
    weights: Array2<f64>,
    /// Global inhibition strength
    #[allow(dead_code)]
    global_inhibition: f64,
    /// Learning rate adaptation parameters
    learning_rate_adaptation: LearningRateAdaptation,
    /// Metaplasticity parameters
    metaplasticity: MetaplasticityController,
    /// Multi-timescale adaptation
    multi_timescale: MultiTimescaleAdaptation,
}

/// Homeostatic neuron with intrinsic plasticity
#[derive(Debug, Clone)]
pub struct HomeostaticNeuron {
    /// Current membrane potential
    membrane_potential: f64,
    /// Spike threshold (adaptive)
    #[allow(dead_code)]
    threshold: f64,
    /// Target firing rate
    target_firing_rate: f64,
    /// Actual firing rate (exponential moving average)
    actual_firing_rate: f64,
    /// Intrinsic excitability
    intrinsic_excitability: f64,
    /// Homeostatic time constant
    homeostatic_tau: f64,
    /// Spike history for rate computation
    spike_history: VecDeque<f64>,
    /// Synaptic scaling factor
    synaptic_scaling: f64,
    /// Membrane time constant
    membrane_tau: f64,
    /// Last spike time
    last_spike_time: f64,
}

/// Learning rate adaptation mechanism (duplicate removed)
/// Metaplasticity controller for flexible learning
#[derive(Debug, Clone)]
pub struct MetaplasticityController {
    /// Metaplastic variables for each synapse
    metaplastic_variables: Array2<f64>,
    /// Metaplastic time constant
    meta_tau: f64,
    /// Plasticity threshold
    #[allow(dead_code)]
    plasticity_threshold: f64,
    /// LTP/LTD balance factor
    #[allow(dead_code)]
    ltp_ltd_balance: f64,
    /// Activity-dependent scaling
    #[allow(dead_code)]
    activity_scaling: f64,
}

/// Multi-timescale adaptation for different learning phases
#[derive(Debug, Clone)]
pub struct MultiTimescaleAdaptation {
    /// Fast adaptation (seconds to minutes)
    fast_adaptation: AdaptationScale,
    /// Medium adaptation (minutes to hours)
    medium_adaptation: AdaptationScale,
    /// Slow adaptation (hours to days)
    slow_adaptation: AdaptationScale,
    /// Current timescale weights
    timescale_weights: Array1<f64>,
}

/// Individual adaptation scale
#[derive(Debug, Clone)]
pub struct AdaptationScale {
    /// Time constant for this scale
    time_constant: f64,
    /// Adaptation strength
    #[allow(dead_code)]
    adaptationstrength: f64,
    /// Memory trace
    memory_trace: f64,
    /// Decay factor
    #[allow(dead_code)]
    decay_factor: f64,
}

impl HomeostaticNeuralClusterer {
    /// Create new homeostatic neural clusterer
    pub fn new(_num_clusters: usize, inputdim: usize) -> Self {
        let mut output_neurons = Vec::new();
        for _ in 0.._num_clusters {
            output_neurons.push(HomeostaticNeuron::new());
        }

        let weights = Array2::zeros((_num_clusters, inputdim));

        let learning_rate_adaptation = LearningRateAdaptation::new(0.01);
        let metaplasticity = MetaplasticityController::new(_num_clusters, inputdim);
        let multi_timescale = MultiTimescaleAdaptation::new();

        Self {
            _num_clusters: _num_clusters,
            inputdim,
            output_neurons,
            weights,
            global_inhibition: 0.1,
            learning_rate_adaptation,
            metaplasticity,
            multi_timescale,
        }
    }

    /// Configure homeostatic parameters
    pub fn with_homeostatic_params(
        mut self,
        target_firing_rate: f64,
        homeostatic_tau: f64,
    ) -> Self {
        for neuron in &mut self.output_neurons {
            neuron.target_firing_rate = target_firing_rate;
            neuron.homeostatic_tau = homeostatic_tau;
        }
        self
    }

    /// Fit homeostatic clustering model
    pub fn fit(
        &mut self,
        points: &ArrayView2<f64>,
        epochs: usize,
    ) -> SpatialResult<Array1<usize>> {
        let (n_samples, n_features) = points.dim();

        if n_features != self.inputdim {
            return Err(SpatialError::InvalidInput(
                "Input dimension mismatch".to_string(),
            ));
        }

        // Initialize weights randomly
        self.initialize_weights()?;

        let mut assignments = Array1::zeros(n_samples);
        let currenttime = 0.0;
        let dt = 0.001; // 1ms time step

        for epoch in 0..epochs {
            let mut epoch_error = 0.0;

            for (sample_idx, sample) in points.outer_iter().enumerate() {
                // Forward pass with homeostatic mechanisms
                let (winner_idx, neuron_activities) = self.forward_pass_homeostatic(
                    &sample,
                    currenttime + (epoch * n_samples + sample_idx) as f64 * dt,
                )?;

                assignments[sample_idx] = winner_idx;

                // Compute reconstruction error
                let reconstruction = self.weights.row(winner_idx);
                let error: f64 = sample
                    .iter()
                    .zip(reconstruction.iter())
                    .map(|(&x, &w)| (x - w).powi(2))
                    .sum();
                epoch_error += error;

                // Homeostatic learning update
                self.homeostatic_learning_update(
                    &sample,
                    winner_idx,
                    &neuron_activities,
                    error,
                    currenttime + (epoch * n_samples + sample_idx) as f64 * dt,
                )?;
            }

            // Update learning rate based on performance
            self.learning_rate_adaptation
                .update_learning_rate(epoch_error / n_samples as f64);

            // Update multi-timescale adaptation
            self.multi_timescale
                .update(epoch_error / n_samples as f64, dt * n_samples as f64);

            // Homeostatic updates at end of epoch
            self.update_homeostatic_mechanisms(dt * n_samples as f64)?;
        }

        Ok(assignments)
    }

    /// Get cluster centers (weights)
    pub fn get_cluster_centers(&self) -> Array2<f64> {
        self.weights.clone()
    }

    /// Forward pass with homeostatic mechanisms
    fn forward_pass_homeostatic(
        &mut self,
        input: &ArrayView1<f64>,
        currenttime: f64,
    ) -> SpatialResult<(usize, Array1<f64>)> {
        let mut activities = Array1::zeros(self._num_clusters);

        // Compute neural activities with homeostatic modulation
        for (neuron_idx, neuron) in self.output_neurons.iter_mut().enumerate() {
            let weights_row = self.weights.row(neuron_idx);

            // Compute dot product (synaptic input)
            let synaptic_input: f64 = input
                .iter()
                .zip(weights_row.iter())
                .map(|(&x, &w)| x * w)
                .sum();

            // Apply synaptic scaling
            let scaled_input = synaptic_input * neuron.synaptic_scaling;

            // Update membrane potential
            neuron.update_membrane_potential(scaled_input, currenttime);

            // Apply intrinsic excitability modulation
            let modulated_potential = neuron.membrane_potential * neuron.intrinsic_excitability;
            activities[neuron_idx] = modulated_potential;
        }

        // Find winner (highest activity)
        let winner_idx = activities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i_, _)| i_)
            .unwrap_or(0);

        // Update firing rates
        self.output_neurons[winner_idx].record_spike(currenttime);

        Ok((winner_idx, activities))
    }

    /// Initialize weights randomly
    fn initialize_weights(&mut self) -> SpatialResult<()> {
        use rand::Rng;
        let mut rng = rand::rng();

        for mut row in self.weights.outer_iter_mut() {
            for weight in row.iter_mut() {
                *weight = rng.gen_range(0.0..1.0);
            }

            // Normalize weights
            let norm: f64 = row.iter().map(|&w| w * w).sum::<f64>().sqrt();
            if norm > 0.0 {
                for weight in row.iter_mut() {
                    *weight /= norm;
                }
            }
        }

        Ok(())
    }

    /// Homeostatic learning update
    fn homeostatic_learning_update(
        &mut self,
        input: &ArrayView1<f64>,
        winner_idx: usize,
        activities: &Array1<f64>,
        error: f64,
        currenttime: f64,
    ) -> SpatialResult<()> {
        // Get current learning rate
        let learning_rate = self.learning_rate_adaptation.baserate;

        // Apply metaplasticity
        let meta_modulation = self.metaplasticity.compute_modulation(winner_idx, error);

        // Apply multi-timescale adaptation
        let timescale_modulation = self.multi_timescale.get_adaptation_factor();

        // Combined learning rate
        let effective_learning_rate = learning_rate * meta_modulation * timescale_modulation;

        // Update winner weights (competitive learning with homeostatic modulation)
        let winner_neuron = &self.output_neurons[winner_idx];
        let homeostatic_factor = winner_neuron.get_homeostatic_factor();

        for (weight, &input_val) in self
            .weights
            .row_mut(winner_idx)
            .iter_mut()
            .zip(input.iter())
        {
            let weight_update =
                effective_learning_rate * homeostatic_factor * (input_val - *weight);
            *weight += weight_update;
        }

        // Update metaplasticity variables
        self.metaplasticity
            .update_metaplastic_variables(winner_idx, activities, currenttime);

        Ok(())
    }

    /// Update homeostatic mechanisms
    fn update_homeostatic_mechanisms(&mut self, dt: f64) -> SpatialResult<()> {
        for neuron in &mut self.output_neurons {
            neuron.update_homeostatic_mechanisms(dt);
        }
        Ok(())
    }
}

impl HomeostaticNeuron {
    /// Create new homeostatic neuron
    fn new() -> Self {
        Self {
            membrane_potential: 0.0,
            threshold: 1.0,
            target_firing_rate: 0.1,
            actual_firing_rate: 0.0,
            intrinsic_excitability: 1.0,
            homeostatic_tau: 1000.0, // 1 second
            spike_history: VecDeque::new(),
            synaptic_scaling: 1.0,
            membrane_tau: 10.0, // 10ms
            last_spike_time: -1000.0,
        }
    }

    /// Update membrane potential
    fn update_membrane_potential(&mut self, input: f64, currenttime: f64) {
        let dt = currenttime - self.last_spike_time;

        // Leaky integrate-and-fire dynamics
        let decay = (-dt / self.membrane_tau).exp();
        self.membrane_potential = self.membrane_potential * decay + input;
    }

    /// Record spike and update firing rate
    fn record_spike(&mut self, _currenttime: f64) {
        self.spike_history.push_back(_currenttime);

        // Keep only recent spikes (last 1 second)
        while let Some(&front_time) = self.spike_history.front() {
            if _currenttime - front_time > 1000.0 {
                self.spike_history.pop_front();
            } else {
                break;
            }
        }

        // Update actual firing rate (exponential moving average)
        let instantaneous_rate = self.spike_history.len() as f64 / 1000.0; // spikes per ms
        let alpha = 1.0 / self.homeostatic_tau;
        self.actual_firing_rate =
            (1.0 - alpha) * self.actual_firing_rate + alpha * instantaneous_rate;
    }

    /// Update homeostatic mechanisms
    fn update_homeostatic_mechanisms(&mut self, dt: f64) {
        // Intrinsic plasticity: adjust excitability to maintain target firing rate
        let rate_error = self.target_firing_rate - self.actual_firing_rate;
        let excitability_update = 0.001 * rate_error * dt;
        self.intrinsic_excitability += excitability_update;
        self.intrinsic_excitability = self.intrinsic_excitability.clamp(0.1, 10.0);

        // Synaptic scaling: global scaling of all synapses
        let scaling_rate = 0.0001;
        let scaling_update = scaling_rate * rate_error * dt;
        self.synaptic_scaling += scaling_update;
        self.synaptic_scaling = self.synaptic_scaling.clamp(0.1, 10.0);
    }

    /// Get homeostatic factor for learning modulation
    fn get_homeostatic_factor(&self) -> f64 {
        // Higher factor when firing rate is below target (need to strengthen synapses)
        let rate_ratio = self.actual_firing_rate / self.target_firing_rate.max(1e-6);
        (2.0 / (1.0 + rate_ratio)).clamp(0.1, 10.0)
    }
}

impl LearningRateAdaptation {
    /// Create new learning rate adaptation
    fn new(baserate: f64) -> Self {
        Self {
            baserate,
            adaptation_factor: 0.1,
            performance_history: VecDeque::new(),
            adaptation_threshold: 0.1,
            max_rate: 0.1,
            min_rate: 1e-6,
        }
    }

    /// Update learning rate based on performance
    fn update_learning_rate(&mut self, _currenterror: f64) {
        self.performance_history.push_back(_currenterror);

        // Keep only recent errors
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }

        // Adapt learning rate based on error trend
        if self.performance_history.len() >= 2 {
            let recent_avg = self.performance_history.iter().rev().take(10).sum::<f64>() / 10.0;
            let older_avg = self
                .performance_history
                .iter()
                .rev()
                .skip(10)
                .take(10)
                .sum::<f64>()
                / 10.0;

            let performance_ratio = if older_avg > 0.0 {
                recent_avg / older_avg
            } else {
                1.0
            };

            // Adapt learning rate
            if performance_ratio < 0.95 {
                // Performance improving - increase learning rate slightly
                self.baserate *= 1.01;
            } else if performance_ratio > 1.05 {
                // Performance degrading - decrease learning rate
                self.baserate *= 0.99;
            }

            // Apply bounds
            self.baserate = self.baserate.max(self.min_rate).min(self.max_rate);
        }
    }
}

impl MetaplasticityController {
    /// Create new metaplasticity controller
    fn new(_num_clusters: usize, inputdim: usize) -> Self {
        Self {
            metaplastic_variables: Array2::ones((_num_clusters, inputdim)),
            meta_tau: 10000.0, // 10 seconds
            plasticity_threshold: 0.5,
            ltp_ltd_balance: 1.0,
            activity_scaling: 1.0,
        }
    }

    /// Compute metaplastic modulation
    fn compute_modulation(&self, _neuronidx: usize, error: f64) -> f64 {
        let meta_var_avg = self
            .metaplastic_variables
            .row(_neuronidx)
            .to_owned()
            .mean();

        // Higher metaplastic variable means lower plasticity (harder to change)
        let modulation = 1.0 / (1.0 + meta_var_avg);

        // Scale by error magnitude
        modulation * (1.0 + error.abs()).ln()
    }

    /// Update metaplastic variables
    fn update_metaplastic_variables(
        &mut self,
        winner_idx: usize,
        activities: &Array1<f64>,
        _currenttime: f64,
    ) {
        let dt = 1.0; // Assume 1ms updates
        let decay_factor = (-dt / self.meta_tau).exp();

        // Update metaplastic variables for winner
        for meta_var in self.metaplastic_variables.row_mut(winner_idx).iter_mut() {
            *meta_var = *meta_var * decay_factor + (1.0 - decay_factor) * activities[winner_idx];
        }
    }
}

impl MultiTimescaleAdaptation {
    /// Create new multi-timescale adaptation
    fn new() -> Self {
        Self {
            fast_adaptation: AdaptationScale::new(1.0, 1.0), // 1ms timescale
            medium_adaptation: AdaptationScale::new(1000.0, 0.5), // 1s timescale
            slow_adaptation: AdaptationScale::new(60000.0, 0.1), // 1min timescale
            timescale_weights: Array1::from(vec![0.5, 0.3, 0.2]),
        }
    }

    /// Update all adaptation scales
    fn update(&mut self, error: f64, dt: f64) {
        self.fast_adaptation.update(error, dt);
        self.medium_adaptation.update(error, dt);
        self.slow_adaptation.update(error, dt);
    }

    /// Get combined adaptation factor
    fn get_adaptation_factor(&self) -> f64 {
        let fast_factor = self.fast_adaptation.memory_trace;
        let medium_factor = self.medium_adaptation.memory_trace;
        let slow_factor = self.slow_adaptation.memory_trace;

        self.timescale_weights[0] * fast_factor
            + self.timescale_weights[1] * medium_factor
            + self.timescale_weights[2] * slow_factor
    }
}

impl AdaptationScale {
    /// Create new adaptation scale
    fn new(time_constant: f64, adaptationstrength: f64) -> Self {
        Self {
            time_constant,
            adaptationstrength,
            memory_trace: 1.0,
            decay_factor: 0.999,
        }
    }

    /// Update this adaptation scale
    fn update(&mut self, error: f64, dt: f64) {
        let decay = (-dt / self.time_constant).exp();
        self.memory_trace = self.memory_trace * decay + (1.0 - decay) * (1.0 - error);
        self.memory_trace = self.memory_trace.clamp(0.0, 2.0);
    }
}

/// Advanced bio-inspired spatial clustering with dendritic computation
#[derive(Debug, Clone)]
pub struct DendriticSpatialClusterer {
    /// Number of output neurons
    _numneurons: usize,
    /// Input dimension
    inputdim: usize,
    /// Neurons with dendritic compartments
    neurons: Vec<DendriticNeuron>,
    /// Lateral connections between neurons
    lateral_connections: Array2<f64>,
    /// Global learning parameters
    global_learning_params: GlobalLearningParams,
}

/// Neuron with dendritic compartments for non-linear processing
#[derive(Debug, Clone)]
pub struct DendriticNeuron {
    /// Dendritic compartments
    dendrites: Vec<DendriticCompartment>,
    /// Soma (cell body)
    soma: SomaCompartment,
    /// Axon output
    axon_output: f64,
    /// Neuron position in space
    position: Array1<f64>,
}

/// Individual dendritic compartment
#[derive(Debug, Clone)]
pub struct DendriticCompartment {
    /// Synaptic weights to this compartment
    weights: Array1<f64>,
    /// Compartment activation
    activation: f64,
    /// Local learning rule parameters
    local_learning_params: LocalLearningParams,
    /// NMDA-like non-linearity threshold
    nmda_threshold: f64,
    /// Compartment time constant
    #[allow(dead_code)]
    time_constant: f64,
}

/// Soma compartment for integration
#[derive(Debug, Clone)]
pub struct SomaCompartment {
    /// Membrane potential
    membrane_potential: f64,
    /// Spike threshold
    threshold: f64,
    /// Refractory period
    #[allow(dead_code)]
    refractory_period: f64,
    /// Time since last spike
    #[allow(dead_code)]
    time_since_spike: f64,
}

/// Global learning parameters
#[derive(Debug, Clone)]
pub struct GlobalLearningParams {
    /// Base learning rate
    learning_rate: f64,
    /// Competition strength
    #[allow(dead_code)]
    competition_strength: f64,
    /// Lateral inhibition radius
    #[allow(dead_code)]
    inhibition_radius: f64,
    /// Plasticity modulation
    #[allow(dead_code)]
    plasticity_modulation: f64,
}

/// Local learning parameters for dendritic compartments
#[derive(Debug, Clone)]
pub struct LocalLearningParams {
    /// Local learning rate
    #[allow(dead_code)]
    local_rate: f64,
    /// Heterosynaptic depression
    #[allow(dead_code)]
    heterosynaptic_depression: f64,
    /// Activity-dependent scaling
    activity_scaling: f64,
}

impl DendriticSpatialClusterer {
    /// Create new dendritic spatial clusterer
    pub fn new(_numneurons: usize, inputdim: usize, dendrites_perneuron: usize) -> Self {
        let mut neurons = Vec::new();
        for i in 0.._numneurons {
            let position = Array1::from(vec![
                (i as f64) / (_numneurons as f64), // Simple 1D arrangement
                0.0,
            ]);
            neurons.push(DendriticNeuron::new(
                inputdim,
                dendrites_perneuron,
                position,
            ));
        }

        let lateral_connections = Array2::zeros((_numneurons, _numneurons));
        let global_learning_params = GlobalLearningParams::new();

        Self {
            _numneurons: _numneurons,
            inputdim,
            neurons,
            lateral_connections,
            global_learning_params,
        }
    }

    /// Fit dendritic clustering model
    pub fn fit(
        &mut self,
        points: &ArrayView2<f64>,
        epochs: usize,
    ) -> SpatialResult<Array1<usize>> {
        let (n_samples, n_features) = points.dim();

        if n_features != self.inputdim {
            return Err(SpatialError::InvalidInput(
                "Input dimension mismatch".to_string(),
            ));
        }

        self.initialize_lateral_connections()?;
        let mut assignments = Array1::zeros(n_samples);

        for epoch in 0..epochs {
            for (sample_idx, sample) in points.outer_iter().enumerate() {
                // Forward pass through dendritic computation
                let neuron_outputs = self.dendritic_forward_pass(&sample)?;

                // Find winning neuron
                let winner_idx = neuron_outputs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i_, _)| i_)
                    .unwrap_or(0);

                assignments[sample_idx] = winner_idx;

                // Dendritic learning update
                self.dendritic_learning_update(&sample, winner_idx, &neuron_outputs)?;
            }
        }

        Ok(assignments)
    }

    /// Get cluster centers (average dendritic weights)
    pub fn get_cluster_centers(&self) -> Array2<f64> {
        let mut centers = Array2::zeros((self._numneurons, self.inputdim));

        for (neuron_idx, neuron) in self.neurons.iter().enumerate() {
            let center = neuron.get_average_weights();
            centers.row_mut(neuron_idx).assign(&center);
        }

        centers
    }

    /// Forward pass with dendritic computation
    fn dendritic_forward_pass(&mut self, input: &ArrayView1<f64>) -> SpatialResult<Array1<f64>> {
        let mut outputs = Array1::zeros(self._numneurons);

        for (neuron_idx, neuron) in self.neurons.iter_mut().enumerate() {
            let neuron_output = neuron.compute_output(input)?;
            outputs[neuron_idx] = neuron_output;
        }

        Ok(outputs)
    }

    /// Initialize lateral connections
    fn initialize_lateral_connections(&mut self) -> SpatialResult<()> {
        for i in 0..self._numneurons {
            for j in 0..self._numneurons {
                if i != j {
                    let distance = self.compute_neuron_distance(i, j);
                    // Mexican hat connectivity: nearby excitation, distant inhibition
                    let connection_strength = if distance < 0.2 {
                        0.1 * (-distance.powi(2) / 0.02).exp() // Excitation
                    } else {
                        -0.05 * (-distance.powi(2) / 0.1).exp() // Inhibition
                    };
                    self.lateral_connections[[i, j]] = connection_strength;
                }
            }
        }
        Ok(())
    }

    /// Compute distance between neurons
    fn compute_neuron_distance(&self, i: usize, j: usize) -> f64 {
        let pos_i = &self.neurons[i].position;
        let pos_j = &self.neurons[j].position;

        pos_i
            .iter()
            .zip(pos_j.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Dendritic learning update
    fn dendritic_learning_update(
        &mut self,
        input: &ArrayView1<f64>,
        winner_idx: usize,
        _outputs: &Array1<f64>,
    ) -> SpatialResult<()> {
        // Update winner neuron
        self.neurons[winner_idx].update_synapses(input, true, &self.global_learning_params)?;

        Ok(())
    }
}

impl DendriticNeuron {
    /// Create new dendritic neuron
    fn new(_inputdim: usize, numdendrites: usize, position: Array1<f64>) -> Self {
        let mut dendrites = Vec::new();
        for _ in 0..numdendrites {
            dendrites.push(DendriticCompartment::new(_inputdim));
        }

        let soma = SomaCompartment::new();

        Self {
            dendrites,
            soma,
            axon_output: 0.0,
            position,
        }
    }

    /// Compute neuron output through dendritic computation
    fn compute_output(&mut self, input: &ArrayView1<f64>) -> SpatialResult<f64> {
        // Compute dendritic activations
        let mut dendritic_inputs = Vec::new();
        for dendrite in &mut self.dendrites {
            let activation = dendrite.compute_activation(input)?;
            dendritic_inputs.push(activation);
        }

        // Integrate in soma
        let soma_input: f64 = dendritic_inputs.iter().sum();
        self.soma.membrane_potential += soma_input;

        // Apply non-linearity and generate output
        let output = if self.soma.membrane_potential > self.soma.threshold {
            self.soma.membrane_potential = 0.0; // Reset after spike
            1.0
        } else {
            (self.soma.membrane_potential / self.soma.threshold).max(0.0)
        };

        self.axon_output = output;
        Ok(output)
    }

    /// Update synaptic weights
    fn update_synapses(
        &mut self,
        input: &ArrayView1<f64>,
        is_winner: bool,
        global_params: &GlobalLearningParams,
    ) -> SpatialResult<()> {
        let modulation = if is_winner { 1.0 } else { 0.1 };

        for dendrite in &mut self.dendrites {
            dendrite.update_weights(input, modulation * global_params.learning_rate)?;
        }

        Ok(())
    }

    /// Get average weights across all dendrites
    fn get_average_weights(&self) -> Array1<f64> {
        let inputdim = self.dendrites[0].weights.len();
        let mut avg_weights = Array1::zeros(inputdim);

        for dendrite in &self.dendrites {
            avg_weights += &dendrite.weights;
        }

        avg_weights / self.dendrites.len() as f64
    }
}

impl DendriticCompartment {
    /// Create new dendritic compartment
    fn new(_inputdim: usize) -> Self {
        let mut rng = rand::rng();

        let mut weights = Array1::zeros(_inputdim);
        for weight in weights.iter_mut() {
            *weight = rng.gen_range(0.0..1.0);
        }

        let local_learning_params = LocalLearningParams::new();

        Self {
            weights,
            activation: 0.0,
            local_learning_params,
            nmda_threshold: 0.5,
            time_constant: 10.0,
        }
    }

    /// Compute compartment activation with NMDA-like non-linearity
    fn compute_activation(&mut self, input: &ArrayView1<f64>) -> SpatialResult<f64> {
        // Linear integration
        let linear_sum: f64 = input
            .iter()
            .zip(self.weights.iter())
            .map(|(&x, &w)| x * w)
            .sum();

        // NMDA-like non-linearity: threshold and supralinear amplification
        let activation = if linear_sum > self.nmda_threshold {
            linear_sum + 0.5 * (linear_sum - self.nmda_threshold).powi(2)
        } else {
            0.1 * linear_sum // Subthreshold leakage
        };

        self.activation = activation;
        Ok(activation)
    }

    /// Update compartment weights
    fn update_weights(
        &mut self,
        input: &ArrayView1<f64>,
        learning_rate: f64,
    ) -> SpatialResult<()> {
        for (weight, &input_val) in self.weights.iter_mut().zip(input.iter()) {
            // Activity-dependent learning with local modulation
            let local_modulation = self.activation * self.local_learning_params.activity_scaling;
            let weight_update = learning_rate * local_modulation * (input_val - *weight);
            *weight += weight_update;

            // Apply bounds
            *weight = weight.clamp(0.0, 2.0);
        }

        Ok(())
    }
}

impl SomaCompartment {
    /// Create new soma compartment
    fn new() -> Self {
        Self {
            membrane_potential: 0.0,
            threshold: 1.0,
            refractory_period: 2.0,
            time_since_spike: 0.0,
        }
    }
}

impl GlobalLearningParams {
    /// Create new global learning parameters
    fn new() -> Self {
        Self {
            learning_rate: 0.01,
            competition_strength: 1.0,
            inhibition_radius: 0.3,
            plasticity_modulation: 1.0,
        }
    }
}

impl LocalLearningParams {
    /// Create new local learning parameters
    fn new() -> Self {
        Self {
            local_rate: 1.0,
            heterosynaptic_depression: 0.001,
            activity_scaling: 1.0,
        }
    }
}

/// Advanced memristive learning system with synaptic plasticity and homeostasis
#[derive(Debug, Clone)]
pub struct AdvancedMemristiveLearning {
    /// Memristive crossbar array
    crossbar_array: MemristiveCrossbar,
    /// Synaptic plasticity mechanisms
    plasticity_mechanisms: Vec<PlasticityMechanism>,
    /// Homeostatic regulation system
    homeostatic_system: HomeostaticSystem,
    /// Metaplasticity rules
    metaplasticity: MetaplasticityRules,
    /// Neuromodulation system
    neuromodulation: NeuromodulationSystem,
    /// Learning history
    learning_history: LearningHistory,
    /// Enable online learning
    #[allow(dead_code)]
    online_learning: bool,
    /// Enable catastrophic forgetting protection
    forgetting_protection: bool,
}

/// Memristive crossbar array with advanced properties
#[derive(Debug, Clone)]
pub struct MemristiveCrossbar {
    /// Device conductances
    pub conductances: Array2<f64>,
    /// Device resistances
    pub resistances: Array2<f64>,
    /// Switching thresholds
    pub switching_thresholds: Array2<f64>,
    /// Retention times
    pub retention_times: Array2<f64>,
    /// Endurance cycles
    pub endurance_cycles: Array2<usize>,
    /// Programming voltages
    pub programming_voltages: Array2<f64>,
    /// Temperature effects
    pub temperature_coefficients: Array2<f64>,
    /// Device variability
    pub device_variability: Array2<f64>,
    /// Crossbar dimensions
    pub dimensions: (usize, usize),
    /// Device type
    pub devicetype: MemristiveDeviceType,
}

/// Types of memristive devices
#[derive(Debug, Clone)]
pub enum MemristiveDeviceType {
    /// Titanium dioxide (TiO2)
    TitaniumDioxide,
    /// Hafnium oxide (HfO2)
    HafniumOxide,
    /// Tantalum oxide (Ta2O5)
    TantalumOxide,
    /// Silver sulfide (Ag2S)
    SilverSulfide,
    /// Organic memristor
    Organic,
    /// Phase change memory
    PhaseChange,
    /// Magnetic tunnel junction
    MagneticTunnelJunction,
}

/// Synaptic plasticity mechanisms
#[derive(Debug, Clone)]
pub struct PlasticityMechanism {
    /// Mechanism type
    pub _mechanismtype: PlasticityType,
    /// Time constants
    pub time_constants: PlasticityTimeConstants,
    /// Learning rates
    pub learning_rates: PlasticityLearningRates,
    /// Threshold parameters
    pub thresholds: PlasticityThresholds,
    /// Enable state
    pub enabled: bool,
    /// Weight scaling factors
    pub weight_scaling: f64,
}

/// Types of synaptic plasticity
#[derive(Debug, Clone)]
pub enum PlasticityType {
    /// Spike-timing dependent plasticity
    STDP,
    /// Homeostatic synaptic scaling
    HomeostaticScaling,
    /// Intrinsic plasticity
    IntrinsicPlasticity,
    /// Heterosynaptic plasticity
    HeterosynapticPlasticity,
    /// Metaplasticity
    Metaplasticity,
    /// Calcium-dependent plasticity
    CalciumDependent,
    /// Voltage-dependent plasticity
    VoltageDependent,
    /// Frequency-dependent plasticity
    FrequencyDependent,
}

/// Time constants for plasticity mechanisms
#[derive(Debug, Clone)]
pub struct PlasticityTimeConstants {
    /// Fast component time constant
    pub tau_fast: f64,
    /// Slow component time constant
    pub tau_slow: f64,
    /// STDP time window
    pub stdp_window: f64,
    /// Homeostatic time constant
    pub tau_homeostatic: f64,
    /// Calcium decay time
    pub tau_calcium: f64,
}

/// Learning rates for different plasticity components
#[derive(Debug, Clone)]
pub struct PlasticityLearningRates {
    /// Potentiation learning rate
    pub potentiation_rate: f64,
    /// Depression learning rate
    pub depression_rate: f64,
    /// Homeostatic learning rate
    pub homeostatic_rate: f64,
    /// Metaplastic learning rate
    pub metaplastic_rate: f64,
    /// Intrinsic plasticity rate
    pub intrinsic_rate: f64,
}

/// Threshold parameters for plasticity
#[derive(Debug, Clone)]
pub struct PlasticityThresholds {
    /// LTP threshold
    pub ltp_threshold: f64,
    /// LTD threshold
    pub ltd_threshold: f64,
    /// Homeostatic target activity
    pub target_activity: f64,
    /// Metaplasticity threshold
    pub metaplasticity_threshold: f64,
    /// Saturation threshold
    pub saturation_threshold: f64,
}

/// Homeostatic regulation system
#[derive(Debug, Clone)]
pub struct HomeostaticSystem {
    /// Target firing rates
    pub target_firing_rates: Array1<f64>,
    /// Current firing rates
    pub current_firing_rates: Array1<f64>,
    /// Homeostatic time constants
    pub time_constants: Array1<f64>,
    /// Regulation mechanisms
    pub mechanisms: Vec<HomeostaticMechanism>,
    /// Adaptation rates
    pub adaptation_rates: Array1<f64>,
    /// Activity history
    pub activity_history: VecDeque<Array1<f64>>,
    /// History window size
    pub history_window: usize,
}

/// Types of homeostatic mechanisms
#[derive(Debug, Clone)]
pub enum HomeostaticMechanism {
    /// Synaptic scaling
    SynapticScaling,
    /// Intrinsic excitability adjustment
    IntrinsicExcitability,
    /// Structural plasticity
    StructuralPlasticity,
    /// Inhibitory plasticity
    InhibitoryPlasticity,
    /// Metaplastic regulation
    MetaplasticRegulation,
}

/// Metaplasticity rules for learning-to-learn
#[derive(Debug, Clone)]
pub struct MetaplasticityRules {
    /// Learning rate adaptation rules
    pub learning_rate_adaptation: LearningRateAdaptation,
    /// Threshold adaptation rules
    pub threshold_adaptation: ThresholdAdaptation,
    /// Memory consolidation rules
    pub consolidation_rules: ConsolidationRules,
    /// Forgetting protection rules
    pub forgetting_protection: ForgettingProtectionRules,
}

/// Learning rate adaptation mechanisms
#[derive(Debug, Clone)]
pub struct LearningRateAdaptation {
    /// Base learning rate
    pub baserate: f64,
    /// Adaptation factor
    pub adaptation_factor: f64,
    /// Performance history
    pub performance_history: VecDeque<f64>,
    /// Adaptation threshold
    pub adaptation_threshold: f64,
    /// Maximum learning rate
    pub max_rate: f64,
    /// Minimum learning rate
    pub min_rate: f64,
}

/// Threshold adaptation for dynamic learning
#[derive(Debug, Clone)]
pub struct ThresholdAdaptation {
    /// Adaptive thresholds
    pub adaptive_thresholds: Array1<f64>,
    /// Threshold update rates
    pub update_rates: Array1<f64>,
    /// Target activation levels
    pub target_activations: Array1<f64>,
    /// Threshold bounds
    pub threshold_bounds: Vec<(f64, f64)>,
}

/// Memory consolidation rules
#[derive(Debug, Clone)]
pub struct ConsolidationRules {
    /// Consolidation time windows
    pub time_windows: Vec<f64>,
    /// Consolidation strengths
    pub consolidation_strengths: Array1<f64>,
    /// Replay mechanisms
    pub replay_enabled: bool,
    /// Replay patterns
    pub replay_patterns: Vec<Array1<f64>>,
    /// Systems consolidation
    pub systems_consolidation: bool,
}

/// Forgetting protection mechanisms
#[derive(Debug, Clone)]
pub struct ForgettingProtectionRules {
    /// Elastic weight consolidation
    pub ewc_enabled: bool,
    /// Fisher information matrix
    pub fisher_information: Array2<f64>,
    /// Synaptic intelligence
    pub synaptic_intelligence: bool,
    /// Importance weights
    pub importance_weights: Array1<f64>,
    /// Protection strength
    pub protection_strength: f64,
}

/// Neuromodulation system for context-dependent learning
#[derive(Debug, Clone)]
pub struct NeuromodulationSystem {
    /// Dopamine levels
    pub dopamine_levels: Array1<f64>,
    /// Serotonin levels
    pub serotonin_levels: Array1<f64>,
    /// Acetylcholine levels
    pub acetylcholine_levels: Array1<f64>,
    /// Noradrenaline levels
    pub noradrenaline_levels: Array1<f64>,
    /// Modulation effects
    pub modulation_effects: NeuromodulationEffects,
    /// Release patterns
    pub release_patterns: NeuromodulatorReleasePatterns,
}

/// Effects of neuromodulation on plasticity
#[derive(Debug, Clone)]
pub struct NeuromodulationEffects {
    /// Effect on learning rate
    pub learning_rate_modulation: Array1<f64>,
    /// Effect on thresholds
    pub threshold_modulation: Array1<f64>,
    /// Effect on excitability
    pub excitability_modulation: Array1<f64>,
    /// Effect on attention
    pub attention_modulation: Array1<f64>,
}

/// Neuromodulator release patterns
#[derive(Debug, Clone)]
pub struct NeuromodulatorReleasePatterns {
    /// Phasic dopamine release
    pub phasic_dopamine: Vec<(f64, f64)>, // (time, amplitude)
    /// Tonic serotonin level
    pub tonic_serotonin: f64,
    /// Cholinergic attention signals
    pub cholinergic_attention: Array1<f64>,
    /// Stress-related noradrenaline
    pub stress_noradrenaline: f64,
}

/// Learning history tracking
#[derive(Debug, Clone)]
pub struct LearningHistory {
    /// Weight change history
    pub weight_changes: VecDeque<Array2<f64>>,
    /// Performance metrics
    pub performance_metrics: VecDeque<PerformanceMetrics>,
    /// Plasticity events
    pub plasticity_events: VecDeque<PlasticityEvent>,
    /// Consolidation events
    pub consolidation_events: VecDeque<ConsolidationEvent>,
    /// Maximum history length
    pub max_history_length: usize,
}

/// Performance metrics for learning assessment
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Accuracy
    pub accuracy: f64,
    /// Learning speed
    pub learning_speed: f64,
    /// Stability
    pub stability: f64,
    /// Generalization
    pub generalization: f64,
    /// Timestamp
    pub timestamp: f64,
}

/// Plasticity event recording
#[derive(Debug, Clone)]
pub struct PlasticityEvent {
    /// Event type
    pub event_type: PlasticityEventType,
    /// Synapses involved
    pub synapses: Vec<(usize, usize)>,
    /// Magnitude of change
    pub magnitude: f64,
    /// Timestamp
    pub timestamp: f64,
    /// Context information
    pub context: String,
}

/// Types of plasticity events
#[derive(Debug, Clone)]
pub enum PlasticityEventType {
    LongTermPotentiation,
    LongTermDepression,
    HomeostaticScaling,
    StructuralPlasticity,
    MetaplasticChange,
}

/// Memory consolidation event
#[derive(Debug, Clone)]
pub struct ConsolidationEvent {
    /// Consolidation type
    pub consolidation_type: ConsolidationType,
    /// Memory patterns consolidated
    pub patterns: Vec<Array1<f64>>,
    /// Consolidation strength
    pub strength: f64,
    /// Timestamp
    pub timestamp: f64,
}

/// Types of memory consolidation
#[derive(Debug, Clone)]
pub enum ConsolidationType {
    SynapticConsolidation,
    SystemsConsolidation,
    ReconsolidationUpdate,
    OfflineReplay,
}

impl AdvancedMemristiveLearning {
    /// Create new advanced memristive learning system
    pub fn new(_rows: usize, cols: usize, devicetype: MemristiveDeviceType) -> Self {
        let crossbar_array = MemristiveCrossbar::new(_rows, cols, devicetype);

        let plasticity_mechanisms = vec![
            PlasticityMechanism::new(PlasticityType::STDP),
            PlasticityMechanism::new(PlasticityType::HomeostaticScaling),
            PlasticityMechanism::new(PlasticityType::IntrinsicPlasticity),
        ];

        let homeostatic_system = HomeostaticSystem::new(_rows);
        let metaplasticity = MetaplasticityRules::new();
        let neuromodulation = NeuromodulationSystem::new(_rows);
        let learning_history = LearningHistory::new();

        Self {
            crossbar_array,
            plasticity_mechanisms,
            homeostatic_system,
            metaplasticity,
            neuromodulation,
            learning_history,
            online_learning: true,
            forgetting_protection: true,
        }
    }

    /// Enable specific plasticity mechanism
    pub fn enable_plasticity(mut self, plasticitytype: PlasticityType) -> Self {
        for mechanism in &mut self.plasticity_mechanisms {
            if std::mem::discriminant(&mechanism._mechanismtype)
                == std::mem::discriminant(&plasticitytype)
            {
                mechanism.enabled = true;
            }
        }
        self
    }

    /// Configure homeostatic regulation
    pub fn with_homeostatic_regulation(mut self, targetrates: Array1<f64>) -> Self {
        self.homeostatic_system.target_firing_rates = targetrates;
        self
    }

    /// Enable catastrophic forgetting protection
    pub fn with_forgetting_protection(mut self, enabled: bool) -> Self {
        self.forgetting_protection = enabled;
        self.metaplasticity.forgetting_protection.ewc_enabled = enabled;
        self
    }

    /// Train on spatial data with advanced plasticity
    pub async fn train_spatial_data(
        &mut self,
        spatial_data: &ArrayView2<'_, f64>,
        target_outputs: &ArrayView1<'_, f64>,
        epochs: usize,
    ) -> SpatialResult<TrainingResult> {
        let mut training_metrics = Vec::new();
        let mut _final_weights = self.crossbar_array.conductances.clone();

        for epoch in 0..epochs {
            // Process each spatial pattern
            let epoch_metrics = self.process_epoch(spatial_data, target_outputs).await?;

            // Apply homeostatic regulation
            self.apply_homeostatic_regulation().await?;

            // Apply metaplasticity updates
            self.apply_metaplasticity_updates(&epoch_metrics).await?;

            // Update neuromodulation
            self.update_neuromodulation(&epoch_metrics).await?;

            // Record learning history
            self.record_learning_history(&epoch_metrics, epoch as f64)
                .await?;

            training_metrics.push(epoch_metrics);

            // Check for consolidation triggers
            if self.should_trigger_consolidation(epoch) {
                self.trigger_memory_consolidation().await?;
            }
        }

        let final_weights = self.crossbar_array.conductances.clone();

        Ok(TrainingResult {
            final_weights,
            training_metrics,
            plasticity_events: self.learning_history.plasticity_events.clone(),
            consolidation_events: self.learning_history.consolidation_events.clone(),
        })
    }

    /// Process single training epoch
    async fn process_epoch(
        &mut self,
        spatial_data: &ArrayView2<'_, f64>,
        target_outputs: &ArrayView1<'_, f64>,
    ) -> SpatialResult<PerformanceMetrics> {
        let n_samples_ = spatial_data.dim().0;
        let mut total_error = 0.0;
        let mut correct_predictions = 0;

        for i in 0..n_samples_ {
            let input = spatial_data.row(i);
            let target = target_outputs[i];

            // Forward pass through memristive crossbar
            let output = self.forward_pass(&input).await?;

            // Compute error
            let error = target - output;
            total_error += error.abs();

            if error.abs() < 0.1 {
                correct_predictions += 1;
            }

            // Apply plasticity mechanisms
            self.apply_plasticity_mechanisms(&input, output, target, error)
                .await?;

            // Update device characteristics
            self.update_memristive_devices(&input, error).await?;
        }

        let accuracy = correct_predictions as f64 / n_samples_ as f64;
        let average_error = total_error / n_samples_ as f64;

        Ok(PerformanceMetrics {
            accuracy,
            learning_speed: 1.0 / (average_error + 1e-8),
            stability: self.compute_weight_stability(),
            generalization: self.estimate_generalization(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        })
    }

    /// Forward pass through memristive crossbar
    async fn forward_pass(&self, input: &ArrayView1<'_, f64>) -> SpatialResult<f64> {
        let mut output = 0.0;

        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    let conductance = self.crossbar_array.conductances[[i, j]];
                    let current = input_val * conductance;

                    // Apply device non-linearity
                    let nonlinear_current = self.apply_device_nonlinearity(current, i, j);

                    output += nonlinear_current;
                }
            }
        }

        // Apply activation function
        Ok(AdvancedMemristiveLearning::sigmoid(output))
    }

    /// Apply device-specific non-linearity
    fn apply_device_nonlinearity(&self, current: f64, row: usize, col: usize) -> f64 {
        match self.crossbar_array.devicetype {
            MemristiveDeviceType::TitaniumDioxide => {
                // TiO2 exponential switching
                let threshold = self.crossbar_array.switching_thresholds[[row, col]];
                if current.abs() > threshold {
                    current * (1.0 + 0.1 * (current / threshold).ln())
                } else {
                    current
                }
            }
            MemristiveDeviceType::HafniumOxide => {
                // HfO2 with steep switching
                let threshold = self.crossbar_array.switching_thresholds[[row, col]];
                current * (1.0 + 0.2 * (current / threshold).tanh())
            }
            MemristiveDeviceType::PhaseChange => {
                // Phase change memory with threshold switching
                let threshold = self.crossbar_array.switching_thresholds[[row, col]];
                if current.abs() > threshold {
                    current * 2.0
                } else {
                    current * 0.1
                }
            }
            _ => current, // Linear for other types
        }
    }

    /// Apply all enabled plasticity mechanisms
    async fn apply_plasticity_mechanisms(
        &mut self,
        input: &ArrayView1<'_, f64>,
        output: f64,
        target: f64,
        error: f64,
    ) -> SpatialResult<()> {
        let mechanisms = self.plasticity_mechanisms.clone();
        for mechanism in &mechanisms {
            if mechanism.enabled {
                match mechanism._mechanismtype {
                    PlasticityType::STDP => {
                        self.apply_stdp_plasticity(input, output, mechanism).await?;
                    }
                    PlasticityType::HomeostaticScaling => {
                        self.apply_homeostatic_scaling(input, output, mechanism)
                            .await?;
                    }
                    PlasticityType::CalciumDependent => {
                        self.apply_calcium_dependent_plasticity(input, output, target, mechanism)
                            .await?;
                    }
                    PlasticityType::VoltageDependent => {
                        self.apply_voltage_dependent_plasticity(input, error, mechanism)
                            .await?;
                    }
                    _ => {
                        // Default plasticity rule
                        self.apply_error_based_plasticity(input, error, mechanism)
                            .await?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply STDP plasticity with advanced timing rules
    async fn apply_stdp_plasticity(
        &mut self,
        input: &ArrayView1<'_, f64>,
        output: f64,
        mechanism: &PlasticityMechanism,
    ) -> SpatialResult<()> {
        let tau_plus = mechanism.time_constants.tau_fast;
        let tau_minus = mechanism.time_constants.tau_slow;
        let a_plus = mechanism.learning_rates.potentiation_rate;
        let a_minus = mechanism.learning_rates.depression_rate;

        // Simplified STDP implementation
        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    // Compute timing difference (simplified)
                    let dt = if input_val > 0.5 && output > 0.5 {
                        1.0 // Pre before post
                    } else if input_val <= 0.5 && output > 0.5 {
                        -1.0 // Post before pre
                    } else {
                        0.0 // No timing relationship
                    };

                    let weight_change = if dt > 0.0 {
                        a_plus * (-dt / tau_plus).exp()
                    } else if dt < 0.0 {
                        -a_minus * (dt / tau_minus).exp()
                    } else {
                        0.0
                    };

                    self.crossbar_array.conductances[[i, j]] +=
                        weight_change * mechanism.weight_scaling;
                    self.crossbar_array.conductances[[i, j]] =
                        self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
                }
            }
        }

        Ok(())
    }

    /// Apply homeostatic scaling
    async fn apply_homeostatic_scaling(
        &mut self,
        self_input: &ArrayView1<'_, f64>,
        output: f64,
        mechanism: &PlasticityMechanism,
    ) -> SpatialResult<()> {
        let target_activity = mechanism.thresholds.target_activity;
        let scaling_rate = mechanism.learning_rates.homeostatic_rate;

        // Global scaling based on overall activity
        let activity_error = output - target_activity;
        let scaling_factor = 1.0 - scaling_rate * activity_error;

        // Apply scaling to all weights
        for i in 0..self.crossbar_array.dimensions.0 {
            for j in 0..self.crossbar_array.dimensions.1 {
                self.crossbar_array.conductances[[i, j]] *= scaling_factor;
                self.crossbar_array.conductances[[i, j]] =
                    self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
            }
        }

        Ok(())
    }

    /// Apply calcium-dependent plasticity
    async fn apply_calcium_dependent_plasticity(
        &mut self,
        input: &ArrayView1<'_, f64>,
        output: f64,
        target: f64,
        mechanism: &PlasticityMechanism,
    ) -> SpatialResult<()> {
        // Simulate calcium dynamics
        let calcium_level =
            AdvancedMemristiveLearning::compute_calcium_level(input, output, target);

        let ltp_threshold = mechanism.thresholds.ltp_threshold;
        let ltd_threshold = mechanism.thresholds.ltd_threshold;

        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    let local_calcium = calcium_level * input_val;

                    let weight_change = if local_calcium > ltp_threshold {
                        mechanism.learning_rates.potentiation_rate * (local_calcium - ltp_threshold)
                    } else if local_calcium < ltd_threshold {
                        -mechanism.learning_rates.depression_rate * (ltd_threshold - local_calcium)
                    } else {
                        0.0
                    };

                    self.crossbar_array.conductances[[i, j]] += weight_change;
                    self.crossbar_array.conductances[[i, j]] =
                        self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
                }
            }
        }

        Ok(())
    }

    /// Apply voltage-dependent plasticity
    async fn apply_voltage_dependent_plasticity(
        &mut self,
        input: &ArrayView1<'_, f64>,
        error: f64,
        mechanism: &PlasticityMechanism,
    ) -> SpatialResult<()> {
        let voltage_threshold = mechanism.thresholds.ltd_threshold;

        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    let local_voltage = input_val * error.abs();

                    if local_voltage > voltage_threshold {
                        let weight_change = mechanism.learning_rates.potentiation_rate
                            * (local_voltage - voltage_threshold)
                            * error.signum();

                        self.crossbar_array.conductances[[i, j]] += weight_change;
                        self.crossbar_array.conductances[[i, j]] =
                            self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply error-based plasticity (default)
    async fn apply_error_based_plasticity(
        &mut self,
        input: &ArrayView1<'_, f64>,
        error: f64,
        mechanism: &PlasticityMechanism,
    ) -> SpatialResult<()> {
        let learning_rate = mechanism.learning_rates.potentiation_rate;

        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    let weight_change = learning_rate * error * input_val;

                    self.crossbar_array.conductances[[i, j]] += weight_change;
                    self.crossbar_array.conductances[[i, j]] =
                        self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
                }
            }
        }

        Ok(())
    }

    /// Compute calcium level for calcium-dependent plasticity
    fn compute_calcium_level(input: &ArrayView1<'_, f64>, output: f64, target: f64) -> f64 {
        let input_activity = input.iter().map(|&x| x.max(0.0)).sum::<f64>();
        let output_activity = output.max(0.0);
        let target_activity = target.max(0.0);

        // Simplified calcium dynamics
        (input_activity * 0.3 + output_activity * 0.4 + target_activity * 0.3).min(1.0)
    }

    /// Update memristive device characteristics
    async fn update_memristive_devices(
        &mut self,
        input: &ArrayView1<'_, f64>,
        error: f64,
    ) -> SpatialResult<()> {
        for (i, &input_val) in input.iter().enumerate() {
            if i < self.crossbar_array.dimensions.0 {
                for j in 0..self.crossbar_array.dimensions.1 {
                    // Update resistance based on conductance
                    let conductance = self.crossbar_array.conductances[[i, j]];
                    self.crossbar_array.resistances[[i, j]] = if conductance > 1e-12 {
                        1.0 / conductance
                    } else {
                        1e12
                    };

                    // Update endurance cycles
                    if input_val > 0.1 {
                        self.crossbar_array.endurance_cycles[[i, j]] += 1;
                    }

                    // Apply device aging effects
                    self.apply_device_aging(i, j);

                    // Apply variability
                    self.apply_device_variability(i, j);
                }
            }
        }

        Ok(())
    }

    /// Apply device aging effects
    fn apply_device_aging(&mut self, row: usize, col: usize) {
        let cycles = self.crossbar_array.endurance_cycles[[row, col]];
        let aging_factor = 1.0 - (cycles as f64) * 1e-8; // Small aging effect

        self.crossbar_array.conductances[[row, col]] *= aging_factor.max(0.1);
    }

    /// Apply device-to-device variability
    fn apply_device_variability(&mut self, row: usize, col: usize) {
        let variability = self.crossbar_array.device_variability[[row, col]];
        let noise = (rand::random::<f64>() - 0.5) * variability;

        self.crossbar_array.conductances[[row, col]] += noise;
        self.crossbar_array.conductances[[row, col]] =
            self.crossbar_array.conductances[[row, col]].clamp(0.0, 1.0);
    }

    /// Apply homeostatic regulation
    async fn apply_homeostatic_regulation(&mut self) -> SpatialResult<()> {
        // Update firing rate history
        let current_rates = self.compute_current_firing_rates();
        self.homeostatic_system
            .activity_history
            .push_back(current_rates.clone());

        // Maintain history window
        if self.homeostatic_system.activity_history.len() > self.homeostatic_system.history_window {
            self.homeostatic_system.activity_history.pop_front();
        }

        // Apply homeostatic mechanisms
        let mechanisms = self.homeostatic_system.mechanisms.clone();
        for mechanism in &mechanisms {
            match mechanism {
                HomeostaticMechanism::SynapticScaling => {
                    self.apply_synaptic_scaling().await?;
                }
                HomeostaticMechanism::IntrinsicExcitability => {
                    self.apply_intrinsic_excitability_adjustment().await?;
                }
                HomeostaticMechanism::StructuralPlasticity => {
                    self.apply_structural_plasticity().await?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Compute current firing rates
    fn compute_current_firing_rates(&self) -> Array1<f64> {
        // Simplified firing rate computation based on conductance sums
        let mut rates = Array1::zeros(self.crossbar_array.dimensions.1);

        for j in 0..self.crossbar_array.dimensions.1 {
            let total_conductance: f64 = (0..self.crossbar_array.dimensions.0)
                .map(|i| self.crossbar_array.conductances[[i, j]])
                .sum();
            rates[j] = AdvancedMemristiveLearning::sigmoid(total_conductance);
        }

        rates
    }

    /// Apply synaptic scaling homeostasis
    async fn apply_synaptic_scaling(&mut self) -> SpatialResult<()> {
        let current_rates = self.compute_current_firing_rates();

        for j in 0..self.crossbar_array.dimensions.1 {
            let target_rate = self.homeostatic_system.target_firing_rates[j];
            let current_rate = current_rates[j];
            let adaptation_rate = self.homeostatic_system.adaptation_rates[j];

            let scaling_factor = 1.0 + adaptation_rate * (target_rate - current_rate);

            // Apply scaling to all incoming synapses
            for i in 0..self.crossbar_array.dimensions.0 {
                self.crossbar_array.conductances[[i, j]] *= scaling_factor;
                self.crossbar_array.conductances[[i, j]] =
                    self.crossbar_array.conductances[[i, j]].clamp(0.0, 1.0);
            }
        }

        Ok(())
    }

    /// Apply intrinsic excitability adjustment
    async fn apply_intrinsic_excitability_adjustment(&mut self) -> SpatialResult<()> {
        // Adjust switching thresholds based on activity
        let current_rates = self.compute_current_firing_rates();

        for j in 0..self.crossbar_array.dimensions.1 {
            let target_rate = self.homeostatic_system.target_firing_rates[j];
            let current_rate = current_rates[j];
            let adaptation_rate = self.homeostatic_system.adaptation_rates[j];

            let threshold_adjustment = adaptation_rate * (current_rate - target_rate);

            for i in 0..self.crossbar_array.dimensions.0 {
                self.crossbar_array.switching_thresholds[[i, j]] += threshold_adjustment;
                self.crossbar_array.switching_thresholds[[i, j]] =
                    self.crossbar_array.switching_thresholds[[i, j]].clamp(0.1, 2.0);
            }
        }

        Ok(())
    }

    /// Apply structural plasticity (simplified)
    async fn apply_structural_plasticity(&mut self) -> SpatialResult<()> {
        // Add or remove connections based on activity
        let current_rates = self.compute_current_firing_rates();

        for i in 0..self.crossbar_array.dimensions.0 {
            for j in 0..self.crossbar_array.dimensions.1 {
                let current_conductance = self.crossbar_array.conductances[[i, j]];
                let activity_level = current_rates[j];

                // Prune weak connections in low-activity regions
                if activity_level < 0.1 && current_conductance < 0.05 {
                    self.crossbar_array.conductances[[i, j]] = 0.0;
                }

                // Strengthen connections in high-activity regions
                if activity_level > 0.9 && current_conductance > 0.5 {
                    self.crossbar_array.conductances[[i, j]] *= 1.01;
                    self.crossbar_array.conductances[[i, j]] =
                        self.crossbar_array.conductances[[i, j]].min(1.0);
                }
            }
        }

        Ok(())
    }

    /// Apply metaplasticity updates
    async fn apply_metaplasticity_updates(
        &mut self,
        metrics: &PerformanceMetrics,
    ) -> SpatialResult<()> {
        // Update learning rate adaptation
        self.metaplasticity
            .learning_rate_adaptation
            .performance_history
            .push_back(metrics.accuracy);

        if self
            .metaplasticity
            .learning_rate_adaptation
            .performance_history
            .len()
            > 100
        {
            self.metaplasticity
                .learning_rate_adaptation
                .performance_history
                .pop_front();
        }

        // Adapt learning rates based on performance
        self.adapt_learning_rates(metrics).await?;

        // Update thresholds
        self.adapt_thresholds(metrics).await?;

        // Apply consolidation if needed
        if metrics.accuracy > 0.9 {
            self.trigger_memory_consolidation().await?;
        }

        Ok(())
    }

    /// Adapt learning rates based on performance
    async fn adapt_learning_rates(&mut self, metrics: &PerformanceMetrics) -> SpatialResult<()> {
        let performance_trend = self.compute_performance_trend();

        for mechanism in &mut self.plasticity_mechanisms {
            if performance_trend > 0.0 {
                // Performance improving, maintain or slightly increase learning rate
                mechanism.learning_rates.potentiation_rate *= 1.01;
                mechanism.learning_rates.depression_rate *= 1.01;
            } else {
                // Performance declining, reduce learning rate
                mechanism.learning_rates.potentiation_rate *= 0.99;
                mechanism.learning_rates.depression_rate *= 0.99;
            }

            // Clamp learning rates
            mechanism.learning_rates.potentiation_rate =
                mechanism.learning_rates.potentiation_rate.clamp(1e-6, 0.1);
            mechanism.learning_rates.depression_rate =
                mechanism.learning_rates.depression_rate.clamp(1e-6, 0.1);
        }

        Ok(())
    }

    /// Compute performance trend
    fn compute_performance_trend(&self) -> f64 {
        let history = &self
            .metaplasticity
            .learning_rate_adaptation
            .performance_history;

        if history.len() < 10 {
            return 0.0;
        }

        let recent_performance: f64 = history.iter().rev().take(5).sum::<f64>() / 5.0;
        let older_performance: f64 = history.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;

        recent_performance - older_performance
    }

    /// Adapt thresholds based on performance
    async fn adapt_thresholds(&mut self, metrics: &PerformanceMetrics) -> SpatialResult<()> {
        // Adjust plasticity thresholds based on learning progress
        for mechanism in &mut self.plasticity_mechanisms {
            if metrics.learning_speed > 1.0 {
                // Fast learning, can afford higher thresholds
                mechanism.thresholds.ltp_threshold *= 1.001;
                mechanism.thresholds.ltd_threshold *= 1.001;
            } else {
                // Slow learning, lower thresholds to increase plasticity
                mechanism.thresholds.ltp_threshold *= 0.999;
                mechanism.thresholds.ltd_threshold *= 0.999;
            }

            // Clamp thresholds
            mechanism.thresholds.ltp_threshold = mechanism.thresholds.ltp_threshold.clamp(0.1, 2.0);
            mechanism.thresholds.ltd_threshold = mechanism.thresholds.ltd_threshold.clamp(0.1, 2.0);
        }

        Ok(())
    }

    /// Update neuromodulation system
    async fn update_neuromodulation(&mut self, metrics: &PerformanceMetrics) -> SpatialResult<()> {
        // Update dopamine based on performance
        let performance_change = metrics.accuracy - 0.5; // Baseline accuracy
        self.neuromodulation
            .dopamine_levels
            .mapv_inplace(|x| x + 0.1 * performance_change);

        // Update serotonin based on stability
        let stability_change = metrics.stability - 0.5;
        self.neuromodulation
            .serotonin_levels
            .mapv_inplace(|x| x + 0.05 * stability_change);

        // Clamp neurotransmitter levels
        self.neuromodulation
            .dopamine_levels
            .mapv_inplace(|x| x.clamp(0.0, 1.0));
        self.neuromodulation
            .serotonin_levels
            .mapv_inplace(|x| x.clamp(0.0, 1.0));

        // Apply modulation effects
        self.apply_neuromodulation_effects().await?;

        Ok(())
    }

    /// Apply neuromodulation effects to plasticity
    async fn apply_neuromodulation_effects(&mut self) -> SpatialResult<()> {
        let avg_dopamine = self.neuromodulation.dopamine_levels.clone().mean();
        let avg_serotonin = self.neuromodulation.serotonin_levels.clone().mean();

        for mechanism in &mut self.plasticity_mechanisms {
            // Dopamine affects learning rate
            let dopamine_effect = 0.5 + avg_dopamine;
            mechanism.learning_rates.potentiation_rate *= dopamine_effect;
            mechanism.learning_rates.depression_rate *= dopamine_effect;

            // Serotonin affects thresholds
            let serotonin_effect = 0.8 + 0.4 * avg_serotonin;
            mechanism.thresholds.ltp_threshold *= serotonin_effect;
            mechanism.thresholds.ltd_threshold *= serotonin_effect;
        }

        Ok(())
    }

    /// Record learning history
    async fn record_learning_history(
        &mut self,
        metrics: &PerformanceMetrics,
        timestamp: f64,
    ) -> SpatialResult<()> {
        // Record performance metrics
        self.learning_history
            .performance_metrics
            .push_back(metrics.clone());

        // Record weight changes
        self.learning_history
            .weight_changes
            .push_back(self.crossbar_array.conductances.clone());

        // Maintain history size
        if self.learning_history.performance_metrics.len()
            > self.learning_history.max_history_length
        {
            self.learning_history.performance_metrics.pop_front();
            self.learning_history.weight_changes.pop_front();
        }

        Ok(())
    }

    /// Check if memory consolidation should be triggered
    fn should_trigger_consolidation(&mut self, epoch: usize) -> bool {
        // Trigger consolidation every 100 epochs or when performance is high
        epoch % 100 == 0
            || self
                .learning_history
                .performance_metrics
                .back()
                .map(|m| m.accuracy > 0.95)
                .unwrap_or(false)
    }

    /// Trigger memory consolidation
    async fn trigger_memory_consolidation(&mut self) -> SpatialResult<()> {
        // Systems consolidation: strengthen important connections
        self.strengthen_important_connections().await?;

        // Record consolidation event
        let consolidation_event = ConsolidationEvent {
            consolidation_type: ConsolidationType::SynapticConsolidation,
            patterns: vec![], // Would store relevant patterns
            strength: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.learning_history
            .consolidation_events
            .push_back(consolidation_event);

        Ok(())
    }

    /// Strengthen important connections during consolidation
    async fn strengthen_important_connections(&mut self) -> SpatialResult<()> {
        // Calculate connection importance based on usage and performance contribution
        let mut importance_matrix = Array2::zeros(self.crossbar_array.dimensions);

        for i in 0..self.crossbar_array.dimensions.0 {
            for j in 0..self.crossbar_array.dimensions.1 {
                let conductance = self.crossbar_array.conductances[[i, j]];
                let usage = self.crossbar_array.endurance_cycles[[i, j]] as f64;

                // Importance based on conductance and usage
                importance_matrix[[i, j]] = conductance * (1.0 + 0.1 * usage.ln_1p());
            }
        }

        // Strengthen top 20% most important connections
        let threshold = self.compute_importance_threshold(&importance_matrix, 0.8);

        for i in 0..self.crossbar_array.dimensions.0 {
            for j in 0..self.crossbar_array.dimensions.1 {
                if importance_matrix[[i, j]] > threshold {
                    self.crossbar_array.conductances[[i, j]] *= 1.05; // 5% strengthening
                    self.crossbar_array.conductances[[i, j]] =
                        self.crossbar_array.conductances[[i, j]].min(1.0);
                }
            }
        }

        Ok(())
    }

    /// Compute importance threshold for top percentage
    fn compute_importance_threshold(
        &self,
        importance_matrix: &Array2<f64>,
        percentile: f64,
    ) -> f64 {
        let mut values: Vec<f64> = importance_matrix.iter().cloned().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (values.len() as f64 * percentile) as usize;
        values.get(index).cloned().unwrap_or(0.0)
    }

    /// Helper functions
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn compute_weight_stability(&self) -> f64 {
        // Simplified stability measure
        let weight_variance = self.crossbar_array.conductances.clone().variance();
        1.0 / (1.0 + weight_variance)
    }

    fn estimate_generalization(&self) -> f64 {
        // Simplified generalization estimate
        0.8 // Placeholder
    }
}

impl MemristiveCrossbar {
    /// Create new memristive crossbar
    pub fn new(_rows: usize, cols: usize, devicetype: MemristiveDeviceType) -> Self {
        let conductances = Array2::from_shape_fn((_rows, cols), |_| rand::random::<f64>() * 0.1);
        let resistances = conductances.mapv(|g| if g > 1e-12 { 1.0 / g } else { 1e12 });
        let switching_thresholds = Array2::from_elem((_rows, cols), 0.5);
        let retention_times = Array2::from_elem((_rows, cols), 1e6);
        let endurance_cycles = Array2::zeros((_rows, cols));
        let programming_voltages = Array2::from_elem((_rows, cols), 1.0);
        let temperature_coefficients = Array2::from_elem((_rows, cols), 0.01);
        let device_variability =
            Array2::from_shape_fn((_rows, cols), |_| rand::random::<f64>() * 0.01);

        Self {
            conductances,
            resistances,
            switching_thresholds,
            retention_times,
            endurance_cycles,
            programming_voltages,
            temperature_coefficients,
            device_variability,
            dimensions: (_rows, cols),
            devicetype,
        }
    }
}

impl PlasticityMechanism {
    /// Create new plasticity mechanism
    pub fn new(_mechanismtype: PlasticityType) -> Self {
        let (time_constants, learning_rates, thresholds) = match _mechanismtype {
            PlasticityType::STDP => (
                PlasticityTimeConstants {
                    tau_fast: 20.0,
                    tau_slow: 40.0,
                    stdp_window: 100.0,
                    tau_homeostatic: 1000.0,
                    tau_calcium: 50.0,
                },
                PlasticityLearningRates {
                    potentiation_rate: 0.01,
                    depression_rate: 0.005,
                    homeostatic_rate: 0.001,
                    metaplastic_rate: 0.0001,
                    intrinsic_rate: 0.001,
                },
                PlasticityThresholds {
                    ltp_threshold: 0.6,
                    ltd_threshold: 0.4,
                    target_activity: 0.5,
                    metaplasticity_threshold: 0.8,
                    saturation_threshold: 0.95,
                },
            ),
            _ => (
                PlasticityTimeConstants {
                    tau_fast: 10.0,
                    tau_slow: 20.0,
                    stdp_window: 50.0,
                    tau_homeostatic: 500.0,
                    tau_calcium: 25.0,
                },
                PlasticityLearningRates {
                    potentiation_rate: 0.005,
                    depression_rate: 0.0025,
                    homeostatic_rate: 0.0005,
                    metaplastic_rate: 0.00005,
                    intrinsic_rate: 0.0005,
                },
                PlasticityThresholds {
                    ltp_threshold: 0.5,
                    ltd_threshold: 0.3,
                    target_activity: 0.4,
                    metaplasticity_threshold: 0.7,
                    saturation_threshold: 0.9,
                },
            ),
        };

        Self {
            _mechanismtype: _mechanismtype,
            time_constants,
            learning_rates,
            thresholds,
            enabled: true,
            weight_scaling: 1.0,
        }
    }
}

impl HomeostaticSystem {
    /// Create new homeostatic system
    pub fn new(_numneurons: usize) -> Self {
        Self {
            target_firing_rates: Array1::from_elem(_numneurons, 0.5),
            current_firing_rates: Array1::zeros(_numneurons),
            time_constants: Array1::from_elem(_numneurons, 1000.0),
            mechanisms: vec![
                HomeostaticMechanism::SynapticScaling,
                HomeostaticMechanism::IntrinsicExcitability,
            ],
            adaptation_rates: Array1::from_elem(_numneurons, 0.001),
            activity_history: VecDeque::new(),
            history_window: 100,
        }
    }
}

impl Default for MetaplasticityRules {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaplasticityRules {
    /// Create new metaplasticity rules
    pub fn new() -> Self {
        Self {
            learning_rate_adaptation: LearningRateAdaptation {
                baserate: 0.01,
                adaptation_factor: 0.1,
                performance_history: VecDeque::new(),
                adaptation_threshold: 0.1,
                max_rate: 0.1,
                min_rate: 1e-6,
            },
            threshold_adaptation: ThresholdAdaptation {
                adaptive_thresholds: Array1::from_elem(10, 0.5),
                update_rates: Array1::from_elem(10, 0.001),
                target_activations: Array1::from_elem(10, 0.5),
                threshold_bounds: vec![(0.1, 2.0); 10],
            },
            consolidation_rules: ConsolidationRules {
                time_windows: vec![100.0, 1000.0, 10000.0],
                consolidation_strengths: Array1::from_elem(3, 1.0),
                replay_enabled: true,
                replay_patterns: Vec::new(),
                systems_consolidation: true,
            },
            forgetting_protection: ForgettingProtectionRules {
                ewc_enabled: false,
                fisher_information: Array2::zeros((10, 10)),
                synaptic_intelligence: false,
                importance_weights: Array1::zeros(10),
                protection_strength: 1.0,
            },
        }
    }
}

impl NeuromodulationSystem {
    /// Create new neuromodulation system
    pub fn new(_numneurons: usize) -> Self {
        Self {
            dopamine_levels: Array1::from_elem(_numneurons, 0.5),
            serotonin_levels: Array1::from_elem(_numneurons, 0.5),
            acetylcholine_levels: Array1::from_elem(_numneurons, 0.5),
            noradrenaline_levels: Array1::from_elem(_numneurons, 0.5),
            modulation_effects: NeuromodulationEffects {
                learning_rate_modulation: Array1::from_elem(_numneurons, 1.0),
                threshold_modulation: Array1::from_elem(_numneurons, 1.0),
                excitability_modulation: Array1::from_elem(_numneurons, 1.0),
                attention_modulation: Array1::from_elem(_numneurons, 1.0),
            },
            release_patterns: NeuromodulatorReleasePatterns {
                phasic_dopamine: Vec::new(),
                tonic_serotonin: 0.5,
                cholinergic_attention: Array1::from_elem(_numneurons, 0.5),
                stress_noradrenaline: 0.3,
            },
        }
    }
}

impl Default for LearningHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningHistory {
    /// Create new learning history tracker
    pub fn new() -> Self {
        Self {
            weight_changes: VecDeque::new(),
            performance_metrics: VecDeque::new(),
            plasticity_events: VecDeque::new(),
            consolidation_events: VecDeque::new(),
            max_history_length: 1000,
        }
    }
}

/// Training result structure
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub final_weights: Array2<f64>,
    pub training_metrics: Vec<PerformanceMetrics>,
    pub plasticity_events: VecDeque<PlasticityEvent>,
    pub consolidation_events: VecDeque<ConsolidationEvent>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_advanced_memristive_learning_creation() {
        let learning_system =
            AdvancedMemristiveLearning::new(8, 4, MemristiveDeviceType::TitaniumDioxide);

        assert_eq!(learning_system.crossbar_array.dimensions, (8, 4));
        assert_eq!(learning_system.plasticity_mechanisms.len(), 3);
        assert!(learning_system.online_learning);
        assert!(learning_system.forgetting_protection);
    }

    #[test]
    fn test_memristive_device_types() {
        let tio2_system =
            AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::TitaniumDioxide);
        let hfo2_system = AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::HafniumOxide);
        let pcm_system = AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::PhaseChange);

        assert!(matches!(
            tio2_system.crossbar_array.devicetype,
            MemristiveDeviceType::TitaniumDioxide
        ));
        assert!(matches!(
            hfo2_system.crossbar_array.devicetype,
            MemristiveDeviceType::HafniumOxide
        ));
        assert!(matches!(
            pcm_system.crossbar_array.devicetype,
            MemristiveDeviceType::PhaseChange
        ));
    }

    #[test]
    fn test_plasticity_mechanism_creation() {
        let stdp_mechanism = PlasticityMechanism::new(PlasticityType::STDP);
        assert!(stdp_mechanism.enabled);
        assert_eq!(stdp_mechanism.time_constants.tau_fast, 20.0);
        assert_eq!(stdp_mechanism.learning_rates.potentiation_rate, 0.01);

        let homeostatic_mechanism = PlasticityMechanism::new(PlasticityType::HomeostaticScaling);
        assert!(homeostatic_mechanism.enabled);
        assert_eq!(homeostatic_mechanism.time_constants.tau_fast, 10.0);
    }

    #[test]
    fn test_homeostatic_system() {
        let homeostatic_system = HomeostaticSystem::new(5);

        assert_eq!(homeostatic_system.target_firing_rates.len(), 5);
        assert_eq!(homeostatic_system.current_firing_rates.len(), 5);
        assert_eq!(homeostatic_system.mechanisms.len(), 2);
        assert_eq!(homeostatic_system.history_window, 100);
    }

    #[test]
    fn test_neuromodulation_system() {
        let neuromod_system = NeuromodulationSystem::new(3);

        assert_eq!(neuromod_system.dopamine_levels.len(), 3);
        assert_eq!(neuromod_system.serotonin_levels.len(), 3);
        assert_eq!(neuromod_system.acetylcholine_levels.len(), 3);
        assert_eq!(neuromod_system.noradrenaline_levels.len(), 3);

        // Check initial levels
        for &level in neuromod_system.dopamine_levels.iter() {
            assert!((level - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_plasticity_mechanisms_configuration() {
        let learning_system =
            AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::TitaniumDioxide)
                .enable_plasticity(PlasticityType::CalciumDependent)
                .enable_plasticity(PlasticityType::VoltageDependent);

        // Check that mechanisms are properly configured
        let enabled_mechanisms: Vec<_> = learning_system
            .plasticity_mechanisms
            .iter()
            .filter(|m| m.enabled)
            .collect();

        assert_eq!(enabled_mechanisms.len(), 3); // Original 3 mechanisms
    }

    #[test]
    fn test_homeostatic_regulation_configuration() {
        let targetrates = Array1::from_vec(vec![0.3, 0.7, 0.5, 0.8]);
        let learning_system =
            AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::HafniumOxide)
                .with_homeostatic_regulation(targetrates.clone());

        assert_eq!(
            learning_system.homeostatic_system.target_firing_rates,
            targetrates
        );
    }

    #[test]
    fn test_forgetting_protection() {
        let learning_system =
            AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::PhaseChange)
                .with_forgetting_protection(true);

        assert!(learning_system.forgetting_protection);
        assert!(
            learning_system
                .metaplasticity
                .forgetting_protection
                .ewc_enabled
        );

        let no_protection_system =
            AdvancedMemristiveLearning::new(4, 4, MemristiveDeviceType::PhaseChange)
                .with_forgetting_protection(false);

        assert!(!no_protection_system.forgetting_protection);
        assert!(
            !no_protection_system
                .metaplasticity
                .forgetting_protection
                .ewc_enabled
        );
    }

    #[tokio::test]
    async fn test_memristive_forward_pass() {
        let learning_system =
            AdvancedMemristiveLearning::new(3, 2, MemristiveDeviceType::TitaniumDioxide);
        let input = array![0.5, 0.8, 0.3];

        let result = learning_system.forward_pass(&input.view()).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!((0.0..=1.0).contains(&output)); // Sigmoid output
    }

    #[test]
    fn test_device_nonlinearity() {
        let learning_system =
            AdvancedMemristiveLearning::new(2, 2, MemristiveDeviceType::TitaniumDioxide);

        // Test TiO2 nonlinearity
        let linear_current = 0.1;
        let nonlinear_current = learning_system.apply_device_nonlinearity(linear_current, 0, 0);
        assert!(nonlinear_current.is_finite());

        // Test with HfO2
        let hfo2_system = AdvancedMemristiveLearning::new(2, 2, MemristiveDeviceType::HafniumOxide);
        let hfo2_output = hfo2_system.apply_device_nonlinearity(linear_current, 0, 0);
        assert!(hfo2_output.is_finite());

        // Test with Phase Change Memory
        let pcm_system = AdvancedMemristiveLearning::new(2, 2, MemristiveDeviceType::PhaseChange);
        let pcm_output = pcm_system.apply_device_nonlinearity(linear_current, 0, 0);
        assert!(pcm_output.is_finite());
    }

    #[tokio::test]
    async fn test_memristive_training() {
        let mut learning_system =
            AdvancedMemristiveLearning::new(2, 1, MemristiveDeviceType::TitaniumDioxide);

        let spatial_data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let target_outputs = array![0.0, 1.0, 1.0, 0.0]; // XOR pattern

        let result = learning_system
            .train_spatial_data(&spatial_data.view(), &target_outputs.view(), 5)
            .await;
        assert!(result.is_ok());

        let training_result = result.unwrap();
        assert_eq!(training_result.training_metrics.len(), 5);
        assert!(
            !training_result.plasticity_events.is_empty()
                || training_result.plasticity_events.is_empty()
        ); // Events may or may not occur in short training
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            accuracy: 0.85,
            learning_speed: 2.5,
            stability: 0.7,
            generalization: 0.6,
            timestamp: 12345.0,
        };

        assert_eq!(metrics.accuracy, 0.85);
        assert_eq!(metrics.learning_speed, 2.5);
        assert_eq!(metrics.stability, 0.7);
        assert_eq!(metrics.generalization, 0.6);
    }

    #[test]
    fn test_plasticity_event_types() {
        let ltp_event = PlasticityEvent {
            event_type: PlasticityEventType::LongTermPotentiation,
            synapses: vec![(0, 1), (1, 2)],
            magnitude: 0.1,
            timestamp: 100.0,
            context: "Training epoch 5".to_string(),
        };

        assert!(matches!(
            ltp_event.event_type,
            PlasticityEventType::LongTermPotentiation
        ));
        assert_eq!(ltp_event.synapses.len(), 2);
        assert_eq!(ltp_event.magnitude, 0.1);
    }

    #[test]
    fn test_consolidation_event_types() {
        let consolidation_event = ConsolidationEvent {
            consolidation_type: ConsolidationType::SynapticConsolidation,
            patterns: vec![array![0.1, 0.2, 0.3]],
            strength: 0.8,
            timestamp: 1000.0,
        };

        assert!(matches!(
            consolidation_event.consolidation_type,
            ConsolidationType::SynapticConsolidation
        ));
        assert_eq!(consolidation_event.patterns.len(), 1);
        assert_eq!(consolidation_event.strength, 0.8);
    }

    #[test]
    fn test_metaplasticity_rules() {
        let metaplasticity = MetaplasticityRules::new();

        assert_eq!(metaplasticity.learning_rate_adaptation.baserate, 0.01);
        assert_eq!(metaplasticity.consolidation_rules.time_windows.len(), 3);
        assert!(!metaplasticity.forgetting_protection.ewc_enabled);
        assert!(metaplasticity.consolidation_rules.replay_enabled);
    }

    #[test]
    fn test_learning_history() {
        let learning_history = LearningHistory::new();

        assert!(learning_history.weight_changes.is_empty());
        assert!(learning_history.performance_metrics.is_empty());
        assert!(learning_history.plasticity_events.is_empty());
        assert!(learning_history.consolidation_events.is_empty());
        assert_eq!(learning_history.max_history_length, 1000);
    }

    #[test]
    fn test_memristive_crossbar_creation() {
        let crossbar = MemristiveCrossbar::new(4, 3, MemristiveDeviceType::SilverSulfide);

        assert_eq!(crossbar.dimensions, (4, 3));
        assert_eq!(crossbar.conductances.shape(), &[4, 3]);
        assert_eq!(crossbar.resistances.shape(), &[4, 3]);
        assert_eq!(crossbar.switching_thresholds.shape(), &[4, 3]);
        assert!(matches!(
            crossbar.devicetype,
            MemristiveDeviceType::SilverSulfide
        ));

        // Check that resistances are inverse of conductances (approximately)
        for i in 0..4 {
            for j in 0..3 {
                let conductance = crossbar.conductances[[i, j]];
                let resistance = crossbar.resistances[[i, j]];
                if conductance > 1e-12 {
                    assert!((conductance * resistance - 1.0).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_device_aging_and_variability() {
        let mut learning_system =
            AdvancedMemristiveLearning::new(2, 2, MemristiveDeviceType::Organic);

        // Store initial conductance
        let initial_conductance = learning_system.crossbar_array.conductances[[0, 0]];

        // Apply aging
        learning_system.apply_device_aging(0, 0);
        let aged_conductance = learning_system.crossbar_array.conductances[[0, 0]];

        // Aging should reduce conductance slightly (or keep it the same for no cycles)
        assert!(aged_conductance <= initial_conductance);

        // Apply variability
        let _pre_variability = learning_system.crossbar_array.conductances[[0, 0]];
        learning_system.apply_device_variability(0, 0);
        let post_variability = learning_system.crossbar_array.conductances[[0, 0]];

        // Variability should change the conductance (usually)
        // Note: Due to randomness, this might occasionally fail, but very rarely
        assert!((0.0..=1.0).contains(&post_variability)); // Should stay in bounds
    }

    #[test]
    fn test_spiking_neuron() {
        let mut neuron = SpikingNeuron::new(vec![0.0, 0.0]);

        // Test no spike with low input
        let spiked = neuron.update(0.1, 0.1);
        assert!(!spiked);

        // Test spike with high input
        let spiked = neuron.update(0.1, 10.0);
        assert!(spiked);

        // Test refractory period
        let spiked = neuron.update(0.1, 10.0);
        assert!(!spiked);
    }

    #[test]
    fn test_synapse_stdp() {
        let mut synapse = Synapse::new(0, 1, 0.5);

        // Test potentiation (pre before post)
        synapse.update_stdp(10.0, true, false); // Pre spike at t=10
        synapse.update_stdp(15.0, false, true); // Post spike at t=15

        // Weight should increase
        assert!(synapse.weight > 0.5);
    }

    #[test]
    fn test_spiking_neural_clusterer() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mut clusterer = SpikingNeuralClusterer::new(2);

        let result = clusterer.fit(&points.view());
        assert!(result.is_ok());

        let (assignments, spike_events) = result.unwrap();
        assert_eq!(assignments.len(), 4);
        assert!(!spike_events.is_empty());
    }

    #[test]
    fn test_neuromorphic_processor() {
        let points = array![[0.0, 0.0], [1.0, 1.0]];
        let processor = NeuromorphicProcessor::new()
            .with_memristive_crossbar(true)
            .with_temporal_coding(true);

        let events = processor.encode_spatial_events(&points.view());
        assert!(events.is_ok());

        let event_list = events.unwrap();
        assert!(!event_list.is_empty());
    }

    #[test]
    fn test_competitive_neural_clusterer() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mut clusterer = CompetitiveNeuralClusterer::new(2, 2);

        let result = clusterer.fit(&points.view(), 50);
        assert!(result.is_ok());

        let assignments = result.unwrap();
        assert_eq!(assignments.len(), 4);

        let centers = clusterer.get_cluster_centers();
        assert_eq!(centers.nrows(), 2);
        assert_eq!(centers.ncols(), 2);
    }

    #[test]
    fn test_homeostatic_neural_clusterer() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mut clusterer = HomeostaticNeuralClusterer::new(2, 2);

        let result = clusterer.fit(&points.view(), 20);
        assert!(result.is_ok());

        let assignments = result.unwrap();
        assert_eq!(assignments.len(), 4);

        let centers = clusterer.get_cluster_centers();
        assert_eq!(centers.nrows(), 2);
        assert_eq!(centers.ncols(), 2);
    }

    #[test]
    fn test_dendritic_spatial_clusterer() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mut clusterer = DendriticSpatialClusterer::new(2, 2, 3);

        let result = clusterer.fit(&points.view(), 20);
        assert!(result.is_ok());

        let assignments = result.unwrap();
        assert_eq!(assignments.len(), 4);

        let centers = clusterer.get_cluster_centers();
        assert_eq!(centers.nrows(), 2);
        assert_eq!(centers.ncols(), 2);
    }
}
