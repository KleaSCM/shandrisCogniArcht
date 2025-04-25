use ndarray::{Array1, Array2, ArrayBase, Ix2, Data, DataMut};
use ndarray_linalg::{Eig, EigVals, QR, SVD};
use std::collections::{HashMap, VecDeque, HashSet};
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn, error};
use chrono::{DateTime, Utc, Duration};
use std::error::Error;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MathEngineError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    
    #[error("Matrix decomposition failed: {0}")]
    DecompositionError(String),
    
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, MathEngineError>;

// Core mathematical structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSpace {
    pub dimensions: usize,
    pub basis: Array2<f32>,  // Orthogonal basis vectors
    pub vectors: HashMap<String, Array1<f32>>,  // Vectors in the space
    epsilon: f32,  // Numerical stability threshold
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionMatrix {
    pub matrix: Array2<f32>,  // Matrix defining interactions
    pub eigenvalues: Array1<f32>,  // Eigenvalues for stability analysis
    pub eigenvectors: Array2<f32>,  // Eigenvectors for trait decomposition
    condition_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovChain {
    pub transition_matrix: Array2<f32>,  // Markov transition matrix
    pub steady_state: Array1<f32>,  // Steady state distribution
    pub convergence_rate: f32,  // Rate of convergence to steady state
}

// Core operations
impl VectorSpace {
    pub fn new(dimensions: usize) -> Result<Self> {
        if dimensions == 0 {
            return Err(MathEngineError::InvalidParameter("Dimensions must be positive".into()));
        }
        
        Ok(Self {
            dimensions,
            basis: Array2::eye(dimensions),
            vectors: HashMap::new(),
            epsilon: 1e-6,
        })
    }

    pub fn add_vector(&mut self, id: String, vector: Array1<f32>) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(MathEngineError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        // Check for NaN or Inf values
        if vector.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(MathEngineError::NumericalInstability(
                "Vector contains NaN or Infinite values".into()
            ));
        }

        // Normalize vector with numerical stability check
        let norm = vector.dot(&vector).sqrt();
        if norm < self.epsilon {
            return Err(MathEngineError::NumericalInstability(
                "Vector magnitude too small for normalization".into()
            ));
        }

        let normalized = vector.mapv(|x| x / norm);
        self.vectors.insert(id, normalized);
        Ok(())
    }

    pub fn project_vector(&self, vector: &Array1<f32>) -> Result<Array1<f32>> {
        if vector.len() != self.dimensions {
            return Err(MathEngineError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }
        Ok(self.basis.t().dot(vector))
    }

    pub fn calculate_similarity(&self, vec1: &str, vec2: &str) -> f32 {
        // Calculate cosine similarity between vectors
        if let (Some(v1), Some(v2)) = (self.vectors.get(vec1), self.vectors.get(vec2)) {
            v1.dot(v2)
        } else {
            0.0
        }
    }
}

impl InteractionMatrix {
    pub fn new(matrix: Array2<f32>) -> Result<Self> {
        let (nrows, ncols) = matrix.dim();
        if nrows != ncols {
            return Err(MathEngineError::InvalidParameter(
                "Interaction matrix must be square".into()
            ));
        }

        // Check for NaN or Inf values
        if matrix.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(MathEngineError::NumericalInstability(
                "Matrix contains NaN or Infinite values".into()
            ));
        }

        // Compute eigendecomposition with error handling
        let (eigenvalues, eigenvectors) = matrix.eig()
            .map_err(|e| MathEngineError::DecompositionError(e.to_string()))?;

        // Check condition number
        let condition_number = Self::calculate_condition_number(&matrix)?;
        if condition_number > 1e6 {
            warn!("High condition number detected: {}", condition_number);
        }

        Ok(Self {
            matrix,
            eigenvalues,
            eigenvectors,
            condition_threshold: 1e6,
        })
    }

    fn calculate_condition_number(matrix: &Array2<f32>) -> Result<f32> {
        let (s, _, _) = matrix.svd(true, true)
            .map_err(|e| MathEngineError::DecompositionError(e.to_string()))?;
        
        let max_singular = s.iter().fold(0.0, |acc, &x| acc.max(x));
        let min_singular = s.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        
        if min_singular < 1e-10 {
            return Err(MathEngineError::NumericalInstability(
                "Matrix is nearly singular".into()
            ));
        }
        
        Ok(max_singular / min_singular)
    }

    pub fn calculate_stability(&self) -> Result<f32> {
        let eigenvalue_stability = self.eigenvalues.iter()
            .try_fold(1.0, |acc, &x| {
                if x.is_nan() || x.is_infinite() {
                    Err(MathEngineError::NumericalInstability(
                        "Invalid eigenvalue detected".into()
                    ))
                } else {
                    Ok(acc.min(x.abs()))
                }
            })?;

        let condition_stability = 1.0 / self.calculate_condition_number(&self.matrix)?;
        Ok(eigenvalue_stability * condition_stability)
    }

    pub fn apply_interaction(&self, vector: &Array1<f32>) -> Result<Array1<f32>> {
        if vector.len() != self.matrix.nrows() {
            return Err(MathEngineError::DimensionMismatch {
                expected: self.matrix.nrows(),
                got: vector.len(),
            });
        }

        let stability = self.calculate_stability()?;
        let interaction = self.matrix.dot(vector);

        // Check for numerical stability in the result
        if interaction.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(MathEngineError::NumericalInstability(
                "Interaction produced invalid values".into()
            ));
        }

        Ok(interaction.mapv(|x| x * stability))
    }

    pub fn decompose_traits(&self) -> HashMap<String, Array1<f32>> {
        // Decompose traits into principal components
        let mut components = HashMap::new();
        for (i, eigenvector) in self.eigenvectors.axis_iter(ndarray::Axis(1)).enumerate() {
            components.insert(format!("component_{}", i), eigenvector.to_owned());
        }
        components
    }
}

impl MarkovChain {
    pub fn new(transition_matrix: Array2<f32>) -> Self {
        // Calculate steady state and convergence rate
        let (eigenvalues, _) = transition_matrix.eig().unwrap();
        let steady_state = Self::calculate_steady_state(&transition_matrix);
        let convergence_rate = Self::calculate_convergence_rate(&eigenvalues);
        
        // Calculate mixing time
        let mixing_time = Self::calculate_mixing_time(&transition_matrix, &steady_state);
        
        Self {
            transition_matrix,
            steady_state,
            convergence_rate,
        }
    }

    fn calculate_steady_state(matrix: &Array2<f32>) -> Array1<f32> {
        // Calculate steady state distribution using power iteration
        let mut current = Array1::ones(matrix.nrows()) / matrix.nrows() as f32;
        let mut previous;
        let tolerance = 1e-6;
        let max_iterations = 1000;
        
        for _ in 0..max_iterations {
            previous = current.clone();
            current = matrix.dot(&current);
            current = current / current.sum();
            
            if (current.clone() - previous).mapv(|x| x.abs()).sum() < tolerance {
                break;
            }
        }
        
        current
    }

    fn calculate_convergence_rate(eigenvalues: &Array1<f32>) -> f32 {
        // Calculate rate of convergence to steady state
        eigenvalues.iter()
            .filter(|&&x| (x - 1.0).abs() > 1e-6)
            .fold(1.0, |acc, &x| acc.min(x.abs()))
    }

    fn calculate_mixing_time(matrix: &Array2<f32>, steady_state: &Array1<f32>) -> usize {
        // Calculate mixing time using total variation distance
        let mut current = Array1::ones(matrix.nrows()) / matrix.nrows() as f32;
        let mut iterations = 0;
        let tolerance = 1e-4;
        let max_iterations = 1000;
        
        while iterations < max_iterations {
            current = matrix.dot(&current);
            let distance = Self::total_variation_distance(&current, steady_state);
            if distance < tolerance {
                break;
            }
            iterations += 1;
        }
        
        iterations
    }

    fn total_variation_distance(dist1: &Array1<f32>, dist2: &Array1<f32>) -> f32 {
        // Calculate total variation distance between two distributions
        0.5 * (dist1.clone() - dist2).mapv(|x| x.abs()).sum()
    }

    pub fn update_state(&self, current_state: &Array1<f32>) -> Array1<f32> {
        // Update state using Markov transition with convergence consideration
        let new_state = self.transition_matrix.dot(current_state);
        let distance_to_steady = Self::total_variation_distance(&new_state, &self.steady_state);
        
        // Adjust transition based on distance to steady state
        if distance_to_steady < 0.1 {
            // Close to steady state, maintain stability
            new_state * 0.9 + self.steady_state.clone() * 0.1
        } else {
            // Far from steady state, allow more change
            new_state
        }
    }

    pub fn calculate_entropy(&self) -> f32 {
        // Calculate entropy of the steady state distribution
        -self.steady_state.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f32>()
    }
}

// Personality system components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitLineage {
    pub trait_id: String,
    pub origin_context: String,
    pub creation_time: DateTime<Utc>,
    pub modification_history: VecDeque<(DateTime<Utc>, String, f32)>, // (timestamp, context, magnitude)
    pub tags: Vec<String>,
    pub parent_traits: Vec<String>,
}

impl TraitLineage {
    pub fn new(trait_id: String, origin_context: String) -> Self {
        Self {
            trait_id,
            origin_context,
            creation_time: Utc::now(),
            modification_history: VecDeque::with_capacity(100), // Keep last 100 modifications
            tags: Vec::new(),
            parent_traits: Vec::new(),
        }
    }

    pub fn add_modification(&mut self, context: String, magnitude: f32) {
        if self.modification_history.len() >= 100 {
            self.modification_history.pop_front();
        }
        self.modification_history.push_back((Utc::now(), context, magnitude));
    }

    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    pub fn add_parent(&mut self, parent_id: String) {
        if !self.parent_traits.contains(&parent_id) {
            self.parent_traits.push(parent_id);
        }
    }

    pub fn get_recent_contexts(&self, duration: Duration) -> Vec<String> {
        let cutoff = Utc::now() - duration;
        self.modification_history
            .iter()
            .filter(|(time, _, _)| *time >= cutoff)
            .map(|(_, context, _)| context.clone())
            .collect()
    }

    pub fn calculate_stability(&self) -> f32 {
        if self.modification_history.len() < 2 {
            return 1.0;
        }

        let mut total_variance = 0.0;
        let mut count = 0;
        
        let mut prev_magnitude = None;
        for (_, _, magnitude) in &self.modification_history {
            if let Some(prev) = prev_magnitude {
                total_variance += (magnitude - prev).abs();
                count += 1;
            }
            prev_magnitude = Some(magnitude);
        }

        if count == 0 {
            return 1.0;
        }

        let avg_variance = total_variance / count as f32;
        1.0 / (1.0 + avg_variance)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitResilience {
    pub base_resilience: Array1<f32>,
    pub current_resilience: Array1<f32>,
    pub recovery_rate: Array1<f32>,
    pub max_resilience: Array1<f32>,
    pub min_resilience: Array1<f32>,
    pub last_update: DateTime<Utc>,
}

impl TraitResilience {
    pub fn new(dimensions: usize) -> Self {
        let base_resilience = Array1::from_elem(dimensions, 0.8); // Start with 80% resilience
        let recovery_rate = Array1::from_elem(dimensions, 0.1); // 10% recovery per time unit
        
        Self {
            base_resilience: base_resilience.clone(),
            current_resilience: base_resilience,
            recovery_rate,
            max_resilience: Array1::ones(dimensions),
            min_resilience: Array1::from_elem(dimensions, 0.1), // Minimum 10% resilience
            last_update: Utc::now(),
        }
    }

    pub fn update(&mut self, stress: &Array1<f32>) -> Result<()> {
        let now = Utc::now();
        let time_delta = (now - self.last_update).num_seconds() as f32;
        
        // Calculate stress impact
        let stress_impact = stress.mapv(|x| x.abs() * time_delta);
        
        // Update current resilience
        self.current_resilience = self.current_resilience.clone() - stress_impact;
        
        // Apply recovery
        let recovery = self.recovery_rate.clone() * time_delta;
        self.current_resilience = self.current_resilience.clone() + recovery;
        
        // Clamp values between min and max
        self.current_resilience = self.current_resilience
            .mapv(|x| x.max(self.min_resilience[0]))
            .mapv(|x| x.min(self.max_resilience[0]));
        
        self.last_update = now;
        Ok(())
    }

    pub fn get_stability_matrix(&self) -> Array2<f32> {
        Array2::from_diag(&self.current_resilience)
    }

    pub fn calculate_hysteresis(&self, trait_vector: &Array1<f32>) -> f32 {
        // Calculate how much the trait vector has changed relative to resilience
        let resilience_weighted = trait_vector * &self.current_resilience;
        resilience_weighted.dot(&resilience_weighted).sqrt()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitInteraction {
    pub source_trait: String,
    pub target_trait: String,
    pub interaction_strength: f32,
    pub interaction_type: InteractionType,
    pub last_activated: DateTime<Utc>,
    pub activation_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Reinforce,    // Positive correlation
    Conflict,     // Negative correlation
    Neutral,      // No significant interaction
    Conditional,  // Depends on context
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitEvolution {
    pub trait_id: String,
    pub generation: u32,
    pub fitness_score: f32,
    pub mutation_rate: f32,
    pub crossover_points: Vec<usize>,
    pub parent_traits: Vec<String>,
    pub selection_pressure: f32,
}

impl TraitEvolution {
    pub fn new(trait_id: String) -> Self {
        Self {
            trait_id,
            generation: 0,
            fitness_score: 1.0,
            mutation_rate: 0.1,
            crossover_points: Vec::new(),
            parent_traits: Vec::new(),
            selection_pressure: 0.5,
        }
    }

    pub fn evolve(&mut self, current_vector: &Array1<f32>, target_vector: &Array1<f32>) -> Result<Array1<f32>> {
        // Calculate fitness based on similarity to target
        let similarity = current_vector.dot(target_vector);
        self.fitness_score = (similarity + 1.0) / 2.0; // Normalize to [0,1]

        // Adjust mutation rate based on fitness
        self.mutation_rate = 0.1 * (1.0 - self.fitness_score);

        // Perform crossover if we have parent traits
        let mut new_vector = if !self.parent_traits.is_empty() {
            self.perform_crossover(current_vector)?
        } else {
            current_vector.clone()
        };

        // Apply mutation
        new_vector = self.apply_mutation(&new_vector)?;

        // Update generation
        self.generation += 1;

        Ok(new_vector)
    }

    fn perform_crossover(&self, current_vector: &Array1<f32>) -> Result<Array1<f32>> {
        let mut result = current_vector.clone();
        
        // Simple one-point crossover
        if let Some(&point) = self.crossover_points.first() {
            if point < result.len() {
                // Swap elements after crossover point
                for i in point..result.len() {
                    result[i] = -result[i]; // Simple inversion for demonstration
                }
            }
        }
        
        Ok(result)
    }

    fn apply_mutation(&self, vector: &Array1<f32>) -> Result<Array1<f32>> {
        let mut result = vector.clone();
        
        // Apply random mutations based on mutation rate
        for i in 0..result.len() {
            if rand::random::<f32>() < self.mutation_rate {
                result[i] += (rand::random::<f32>() - 0.5) * 0.2; // Small random adjustment
            }
        }
        
        // Normalize the result
        let norm = result.dot(&result).sqrt();
        if norm < 1e-10 {
            return Err(MathEngineError::NumericalInstability(
                "Mutation produced zero vector".into()
            ));
        }
        
        Ok(result.mapv(|x| x / norm))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitCluster {
    pub centroid: Array1<f32>,
    pub member_traits: Vec<String>,
    pub cohesion: f32,
    pub separation: f32,
    pub silhouette_score: f32,
}

impl TraitCluster {
    pub fn new(centroid: Array1<f32>) -> Self {
        Self {
            centroid,
            member_traits: Vec::new(),
            cohesion: 0.0,
            separation: 0.0,
            silhouette_score: 0.0,
        }
    }

    pub fn update_metrics(&mut self, all_traits: &HashMap<String, Array1<f32>>) -> Result<()> {
        // Calculate cohesion (average distance to centroid)
        let mut total_distance = 0.0;
        for trait_id in &self.member_traits {
            if let Some(vector) = all_traits.get(trait_id) {
                let distance = 1.0 - vector.dot(&self.centroid);
                total_distance += distance;
            }
        }
        
        self.cohesion = if !self.member_traits.is_empty() {
            total_distance / self.member_traits.len() as f32
        } else {
            0.0
        };

        // Calculate separation (distance to nearest other cluster)
        // This would typically be calculated across all clusters
        // For now, we'll use a placeholder
        self.separation = 1.0 - self.cohesion;

        // Calculate silhouette score
        self.silhouette_score = (self.separation - self.cohesion) / 
            self.cohesion.max(self.separation);

        Ok(())
    }

    pub fn add_member(&mut self, trait_id: String, vector: &Array1<f32>) -> Result<()> {
        self.member_traits.push(trait_id);
        
        // Update centroid
        let n = self.member_traits.len() as f32;
        self.centroid = (&self.centroid * (n - 1.0) + vector) / n;
        
        // Normalize centroid
        let norm = self.centroid.dot(&self.centroid).sqrt();
        if norm < 1e-10 {
            return Err(MathEngineError::NumericalInstability(
                "Cluster centroid has zero magnitude".into()
            ));
        }
        
        self.centroid = self.centroid.mapv(|x| x / norm);
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitGoal {
    pub goal_id: String,
    pub target_vector: Array1<f32>,
    pub priority: f32,
    pub deadline: Option<DateTime<Utc>>,
    pub progress: f32,
    pub subgoals: Vec<TraitGoal>,
}

impl TraitGoal {
    pub fn new(goal_id: String, target_vector: Array1<f32>, priority: f32) -> Self {
        Self {
            goal_id,
            target_vector,
            priority,
            deadline: None,
            progress: 0.0,
            subgoals: Vec::new(),
        }
    }

    pub fn calculate_progress(&self, current_vector: &Array1<f32>) -> f32 {
        let similarity = current_vector.dot(&this.target_vector);
        (similarity + 1.0) / 2.0 // Normalize to [0,1]
    }

    pub fn update_progress(&mut self, current_vector: &Array1<f32>) {
        this.progress = this.calculate_progress(current_vector);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitMemory {
    pub memory_id: String,
    pub vector: Array1<f32>,
    pub strength: f32,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    pub decay_rate: f32,
    pub consolidation_strength: f32,
}

impl TraitMemory {
    pub fn new(memory_id: String, vector: Array1<f32>) -> Self {
        Self {
            memory_id,
            vector,
            strength: 1.0,
            last_accessed: Utc::now(),
            access_count: 1,
            decay_rate: 0.01,
            consolidation_strength: 0.0,
        }
    }

    pub fn update_strength(&mut self) {
        let time_since_access = (Utc::now() - this.last_accessed).num_seconds() as f32;
        let decay = this.decay_rate * time_since_access;
        this.strength = (this.strength - decay).max(0.0);
        this.last_accessed = Utc::now();
        this.access_count += 1;
        
        // Increase consolidation strength with repeated access
        this.consolidation_strength = (this.consolidation_strength + 0.1).min(1.0);
    }

    pub fn should_forget(&self) -> bool {
        this.strength < 0.1 && this.consolidation_strength < 0.3
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitConflict {
    pub conflicting_traits: Vec<String>,
    pub conflict_strength: f32,
    pub resolution_history: Vec<(DateTime<Utc>, String, f32)>, // (timestamp, resolution_type, effectiveness)
    pub current_mediation: Option<String>,
}

impl TraitConflict {
    pub fn new(conflicting_traits: Vec<String>) -> Self {
        Self {
            conflicting_traits,
            conflict_strength: 1.0,
            resolution_history: Vec::new(),
            current_mediation: None,
        }
    }

    pub fn calculate_conflict_strength(&self, traits: &HashMap<String, Array1<f32>>) -> Result<f32> {
        if this.conflicting_traits.len() < 2 {
            return Ok(0.0);
        }

        let mut total_conflict = 0.0;
        let mut count = 0;

        for i in 0..this.conflicting_traits.len() {
            for j in (i + 1)..this.conflicting_traits.len() {
                if let (Some(vec1), Some(vec2)) = (
                    traits.get(&this.conflicting_traits[i]),
                    traits.get(&this.conflicting_traits[j])
                ) {
                    let similarity = vec1.dot(vec2);
                    total_conflict += 1.0 - similarity;
                    count += 1;
                }
            }
        }

        Ok(if count > 0 { total_conflict / count as f32 } else { 0.0 })
    }

    pub fn add_resolution(&mut self, resolution_type: String, effectiveness: f32) {
        this.resolution_history.push((Utc::now(), resolution_type, effectiveness));
        this.conflict_strength *= (1.0 - effectiveness);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitAdaptation {
    pub learning_rate: f32,
    pub adaptation_strength: f32,
    pub feedback_history: VecDeque<(DateTime<Utc>, f32)>, // (timestamp, feedback)
    pub success_rate: f32,
    pub adaptation_threshold: f32,
}

impl TraitAdaptation {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.1,
            adaptation_strength: 1.0,
            feedback_history: VecDeque::with_capacity(100),
            success_rate: 0.5,
            adaptation_threshold: 0.7,
        }
    }

    pub fn update_learning_rate(&mut self) {
        // Decay learning rate based on success rate
        let decay_factor = 1.0 - (self.success_rate * 0.1);
        self.learning_rate = (self.learning_rate * decay_factor).max(0.01);
    }

    pub fn add_feedback(&mut self, feedback: f32) {
        if self.feedback_history.len() >= 100 {
            self.feedback_history.pop_front();
        }
        self.feedback_history.push_back((Utc::now(), feedback));
        
        // Update success rate
        let positive_feedback = self.feedback_history.iter()
            .filter(|(_, f)| *f > 0.0)
            .count() as f32;
        self.success_rate = positive_feedback / self.feedback_history.len() as f32;
    }

    pub fn should_adapt(&self) -> bool {
        self.success_rate < self.adaptation_threshold
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitHierarchy {
    pub parent_trait: String,
    pub child_traits: Vec<String>,
    pub inheritance_strength: f32,
    pub specialization_factor: f32,
    pub hierarchy_depth: usize,
}

impl TraitHierarchy {
    pub fn new(parent_trait: String) -> Self {
        Self {
            parent_trait,
            child_traits: Vec::new(),
            inheritance_strength: 0.8,
            specialization_factor: 0.2,
            hierarchy_depth: 1,
        }
    }

    pub fn add_child(&mut self, child_trait: String) {
        if !self.child_traits.contains(&child_trait) {
            self.child_traits.push(child_trait);
        }
    }

    pub fn calculate_inheritance(&self, parent_vector: &Array1<f32>) -> Array1<f32> {
        // Calculate inherited traits with specialization
        let mut inherited = parent_vector.clone() * self.inheritance_strength;
        let specialization = Array1::from_shape_fn(parent_vector.len(), |_| {
            (rand::random::<f32>() - 0.5) * self.specialization_factor
        });
        inherited + specialization
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitContext {
    pub context_id: String,
    pub activation_vector: Array1<f32>,
    pub relevance_scores: HashMap<String, f32>,
    pub context_strength: f32,
    pub last_activated: DateTime<Utc>,
}

impl TraitContext {
    pub fn new(context_id: String, dimensions: usize) -> Self {
        Self {
            context_id,
            activation_vector: Array1::zeros(dimensions),
            relevance_scores: HashMap::new(),
            context_strength: 1.0,
            last_activated: Utc::now(),
        }
    }

    pub fn update_relevance(&mut self, trait_id: &str, relevance: f32) {
        self.relevance_scores.insert(trait_id.to_string(), relevance);
    }

    pub fn calculate_activation(&self, trait_vector: &Array1<f32>) -> f32 {
        // Calculate how much this context activates the trait
        let similarity = self.activation_vector.dot(trait_vector);
        let relevance = self.relevance_scores.get(trait_id)
            .unwrap_or(&0.0);
        similarity * relevance * self.context_strength
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitEngine {
    pub trait_space: VectorSpace,
    pub interaction_matrix: InteractionMatrix,
    pub resilience: TraitResilience,
    pub lineages: HashMap<String, TraitLineage>,
    pub interactions: HashMap<String, TraitInteraction>,
    pub evolution: HashMap<String, TraitEvolution>,
    pub clusters: Vec<TraitCluster>,
    pub memories: HashMap<String, TraitMemory>,
    pub adaptations: HashMap<String, TraitAdaptation>,
}

impl TraitEngine {
    pub fn new(dimensions: usize) -> Result<Self> {
        // Initialize with random interaction matrix
        let interaction_matrix = Array2::from_shape_fn((dimensions, dimensions), |_| rand::random::<f32>());
        
        Ok(Self {
            trait_space: VectorSpace::new(dimensions)?,
            interaction_matrix: InteractionMatrix::new(interaction_matrix)?,
            resilience: TraitResilience::new(dimensions),
            lineages: HashMap::new(),
            interactions: HashMap::new(),
            evolution: HashMap::new(),
            clusters: Vec::new(),
            memories: HashMap::new(),
            adaptations: HashMap::new(),
        })
    }

    pub fn update_trait(&mut self, trait_id: &str, new_vector: Array1<f32>, context: &str) -> Result<()> {
        // Update trait vector with stability consideration
        if let Some(current_vector) = self.trait_space.vectors.get(trait_id) {
            let interaction = self.interaction_matrix.apply_interaction(&new_vector)?;
            
            // Get current stability matrix from resilience
            let stability_matrix = self.resilience.get_stability_matrix();
            
            // Calculate change magnitude for lineage tracking
            let change_magnitude = (interaction.clone() - current_vector)
                .mapv(|x| x.abs())
                .sum();
            
            let stabilized = stability_matrix.dot(current_vector) + 
                           (Array2::eye(self.trait_space.dimensions) - &stability_matrix).dot(&interaction);
            
            self.trait_space.add_vector(trait_id.to_string(), stabilized)?;
            
            // Update lineage
            self.update_lineage(trait_id, context, change_magnitude)?;
            
            // Update resilience based on the change
            self.resilience.update(&(interaction - current_vector))?;
        } else {
            // Create new trait with lineage
            self.trait_space.add_vector(trait_id.to_string(), new_vector)?;
            let lineage = TraitLineage::new(trait_id.to_string(), context.to_string());
            self.lineages.insert(trait_id.to_string(), lineage);
        }
        
        Ok(())
    }

    fn update_lineage(&mut self, trait_id: &str, context: &str, magnitude: f32) -> Result<()> {
        if let Some(lineage) = self.lineages.get_mut(trait_id) {
            lineage.add_modification(context.to_string(), magnitude);
        } else {
            return Err(MathEngineError::InvalidParameter(
                format!("No lineage found for trait {}", trait_id)
            ));
        }
        Ok(())
    }

    pub fn get_trait_stability(&self, trait_id: &str) -> Result<f32> {
        if let Some(lineage) = self.lineages.get(trait_id) {
            let lineage_stability = lineage.calculate_stability();
            let resilience_stability = self.resilience.calculate_hysteresis(
                self.trait_space.vectors.get(trait_id)
                    .ok_or_else(|| MathEngineError::InvalidParameter("Trait not found".into()))?
            );
            
            Ok(lineage_stability * resilience_stability)
        } else {
            Err(MathEngineError::InvalidParameter(
                format!("No lineage found for trait {}", trait_id)
            ))
        }
    }

    pub fn add_trait_tag(&mut self, trait_id: &str, tag: String) -> Result<()> {
        if let Some(lineage) = self.lineages.get_mut(trait_id) {
            lineage.add_tag(tag);
            Ok(())
        } else {
            Err(MathEngineError::InvalidParameter(
                format!("No lineage found for trait {}", trait_id)
            ))
        }
    }

    pub fn get_trait_contexts(&self, trait_id: &str, duration: Duration) -> Result<Vec<String>> {
        if let Some(lineage) = self.lineages.get(trait_id) {
            Ok(lineage.get_recent_contexts(duration))
        } else {
            Err(MathEngineError::InvalidParameter(
                format!("No lineage found for trait {}", trait_id)
            ))
        }
    }

    pub fn add_interaction(&mut self, source: &str, target: &str, strength: f32, interaction_type: InteractionType) -> Result<()> {
        let interaction = TraitInteraction {
            source_trait: source.to_string(),
            target_trait: target.to_string(),
            interaction_strength: strength,
            interaction_type,
            last_activated: Utc::now(),
            activation_count: 0,
        };
        
        let key = format!("{}:{}", source, target);
        self.interactions.insert(key, interaction);
        Ok(())
    }

    pub fn update_trait_with_evolution(&mut self, trait_id: &str, target_vector: &Array1<f32>, context: &str) -> Result<()> {
        if let Some(current_vector) = self.trait_space.vectors.get(trait_id) {
            // Get or create evolution tracker
            let evolution = self.evolution.entry(trait_id.to_string())
                .or_insert_with(|| TraitEvolution::new(trait_id.to_string()));
            
            // Evolve the trait
            let evolved_vector = evolution.evolve(current_vector, target_vector)?;
            
            // Update trait with evolved vector
            self.update_trait(trait_id, evolved_vector, context)?;
            
            // Update clusters
            self.update_clusters(trait_id, &evolved_vector)?;
        }
        Ok(())
    }

    fn update_clusters(&mut self, trait_id: &str, vector: &Array1<f32>) -> Result<()> {
        // Find nearest cluster
        let mut best_cluster_index = None;
        let mut best_similarity = -1.0;
        
        for (i, cluster) in self.clusters.iter().enumerate() {
            let similarity = vector.dot(&cluster.centroid);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_cluster_index = Some(i);
            }
        }
        
        // Add to best cluster or create new one
        if let Some(index) = best_cluster_index {
            self.clusters[index].add_member(trait_id.to_string(), vector)?;
        } else {
            let new_cluster = TraitCluster::new(vector.clone());
            self.clusters.push(new_cluster);
        }
        
        // Update cluster metrics
        for cluster in &mut self.clusters {
            cluster.update_metrics(&self.trait_space.vectors)?;
        }
        
        Ok(())
    }

    pub fn get_similar_traits(&self, trait_id: &str, threshold: f32) -> Result<Vec<(String, f32)>> {
        let vector = self.trait_space.vectors.get(trait_id)
            .ok_or_else(|| MathEngineError::InvalidParameter("Trait not found".into()))?;
        
        let mut similarities = Vec::new();
        
        for (other_id, other_vector) in &self.trait_space.vectors {
            if other_id != trait_id {
                let similarity = vector.dot(other_vector);
                if similarity >= threshold {
                    similarities.push((other_id.clone(), similarity));
                }
            }
        }
        
        // Sort by similarity
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(similarities)
    }

    pub fn get_trait_cluster(&self, trait_id: &str) -> Result<Option<&TraitCluster>> {
        for cluster in &self.clusters {
            if cluster.member_traits.contains(&trait_id.to_string()) {
                return Ok(Some(cluster));
            }
        }
        Ok(None)
    }

    pub fn propagate_influence(&mut self, source_trait: &str, strength: f32, depth: usize) -> Result<()> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((source_trait.to_string(), strength, 0));

        while let Some((current_trait, current_strength, current_depth)) = queue.pop_front() {
            if current_depth > depth || visited.contains(&current_trait) {
                continue;
            }
            visited.insert(current_trait.clone());

            // Get all interactions for current trait
            for (key, interaction) in &this.interactions {
                if interaction.source_trait == current_trait {
                    let next_trait = &interaction.target_trait;
                    let next_strength = current_strength * interaction.interaction_strength;

                    // Apply influence
                    if let Some(vector) = this.trait_space.vectors.get_mut(next_trait) {
                        let influence = vector * next_strength;
                        *vector = (vector + influence).normalize()?;
                    }

                    queue.push_back((next_trait.clone(), next_strength, current_depth + 1));
                }
            }
        }
        Ok(())
    }

    pub fn consolidate_memories(&mut self) -> Result<()> {
        let mut to_remove = Vec::new();
        
        for (memory_id, memory) in &mut this.memories {
            memory.update_strength();
            
            if memory.should_forget() {
                to_remove.push(memory_id.clone());
            }
        }
        
        for memory_id in to_remove {
            this.memories.remove(&memory_id);
        }
        
        Ok(())
    }

    pub fn resolve_conflicts(&mut self) -> Result<()> {
        let mut conflicts = this.identify_conflicts()?;
        
        for conflict in &mut conflicts {
            let conflict_strength = conflict.calculate_conflict_strength(&this.trait_space.vectors)?;
            
            if conflict_strength > 0.5 {
                // Try to resolve conflict
                let resolution = this.mediate_conflict(conflict)?;
                conflict.add_resolution("mediation".to_string(), resolution);
            }
        }
        
        Ok(())
    }

    fn identify_conflicts(&self) -> Result<Vec<TraitConflict>> {
        let mut conflicts = Vec::new();
        let mut visited = HashSet::new();
        
        for (trait_id, vector) in &this.trait_space.vectors {
            if visited.contains(trait_id) {
                continue;
            }
            
            let mut conflicting_traits = Vec::new();
            
            for (other_id, other_vector) in &this.trait_space.vectors {
                if trait_id != other_id {
                    let similarity = vector.dot(other_vector);
                    if similarity < -0.5 { // Strong negative correlation
                        conflicting_traits.push(other_id.clone());
                        visited.insert(other_id.clone());
                    }
                }
            }
            
            if !conflicting_traits.is_empty() {
                conflicting_traits.push(trait_id.clone());
                conflicts.push(TraitConflict::new(conflicting_traits));
            }
        }
        
        Ok(conflicts)
    }

    fn mediate_conflict(&self, conflict: &TraitConflict) -> Result<f32> {
        // Simple mediation: average the conflicting vectors
        let mut combined_vector = Array1::zeros(this.trait_space.dimensions);
        let mut count = 0;
        
        for trait_id in &conflict.conflicting_traits {
            if let Some(vector) = this.trait_space.vectors.get(trait_id) {
                combined_vector = combined_vector + vector;
                count += 1;
            }
        }
        
        if count > 0 {
            combined_vector = combined_vector / count as f32;
            let norm = combined_vector.dot(&combined_vector).sqrt();
            if norm > 1e-10 {
                combined_vector = combined_vector.mapv(|x| x / norm);
                
                // Calculate effectiveness as average similarity to combined vector
                let mut total_similarity = 0.0;
                for trait_id in &conflict.conflicting_traits {
                    if let Some(vector) = this.trait_space.vectors.get(trait_id) {
                        total_similarity += vector.dot(&combined_vector);
                    }
                }
                
                return Ok(total_similarity / conflict.conflicting_traits.len() as f32);
            }
        }
        
        Ok(0.0)
    }

    pub fn pursue_goal(&mut self, goal: &TraitGoal) -> Result<()> {
        // Calculate current progress
        let current_vector = this.trait_space.vectors.get(&goal.goal_id)
            .ok_or_else(|| MathEngineError::InvalidParameter("Trait not found".into()))?;
        
        let progress = goal.calculate_progress(current_vector);
        
        if progress < 0.9 { // If not close enough to goal
            // Evolve trait towards goal
            this.update_trait_with_evolution(&goal.goal_id, &goal.target_vector, "goal_pursuit")?;
            
            // Propagate influence to related traits
            this.propagate_influence(&goal.goal_id, goal.priority, 2)?;
        }
        
        Ok(())
    }

    pub fn add_goal(&mut self, goal: TraitGoal) -> Result<()> {
        // Initialize trait if it doesn't exist
        if !this.trait_space.vectors.contains_key(&goal.goal_id) {
            this.trait_space.add_vector(
                goal.goal_id.clone(),
                Array1::zeros(this.trait_space.dimensions)
            )?;
        }
        
        // Start pursuing the goal
        this.pursue_goal(&goal)?;
        
        Ok(())
    }

    pub fn adapt_trait(&mut self, trait_id: &str, feedback: f32) -> Result<()> {
        let adaptation = self.adaptations.entry(trait_id.to_string())
            .or_insert_with(TraitAdaptation::new);
        
        adaptation.add_feedback(feedback);
        
        if adaptation.should_adapt() {
            // Get current vector
            let current_vector = self.trait_space.vectors.get(trait_id)
                .ok_or_else(|| MathEngineError::InvalidParameter("Trait not found".into()))?;
            
            // Calculate adaptation direction based on feedback
            let adaptation_direction = if feedback > 0.0 {
                current_vector.clone()
            } else {
                -current_vector.clone()
            };
            
            // Apply adaptation
            let adapted_vector = current_vector + 
                adaptation_direction * adaptation.learning_rate * adaptation.adaptation_strength;
            
            // Update trait
            self.update_trait(trait_id, adapted_vector, "adaptation")?;
            
            // Update learning rate
            adaptation.update_learning_rate();
        }
        
        Ok(())
    }

    pub fn create_hierarchy(&mut self, parent_trait: &str, child_traits: Vec<String>) -> Result<()> {
        let hierarchy = TraitHierarchy::new(parent_trait.to_string());
        
        // Get parent vector
        let parent_vector = self.trait_space.vectors.get(parent_trait)
            .ok_or_else(|| MathEngineError::InvalidParameter("Parent trait not found".into()))?;
        
        // Create child traits with inheritance
        for child_id in child_traits {
            let inherited_vector = hierarchy.calculate_inheritance(parent_vector);
            self.update_trait(&child_id, inherited_vector, "inheritance")?;
            hierarchy.add_child(child_id);
        }
        
        Ok(())
    }

    pub fn add_context(&mut self, context: TraitContext) -> Result<()> {
        // Store context activation vector
        self.trait_space.add_vector(
            format!("context_{}", context.context_id),
            context.activation_vector
        )?;
        
        Ok(())
    }

    pub fn activate_traits_in_context(&mut self, context_id: &str) -> Result<()> {
        // Get context
        let context_vector = self.trait_space.vectors.get(&format!("context_{}", context_id))
            .ok_or_else(|| MathEngineError::InvalidParameter("Context not found".into()))?;
        
        // Activate relevant traits
        for (trait_id, trait_vector) in &self.trait_space.vectors {
            if !trait_id.starts_with("context_") {
                let activation = context_vector.dot(trait_vector);
                if activation > 0.7 { // High activation threshold
                    // Boost trait influence
                    self.propagate_influence(trait_id, activation, 2)?;
                }
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodEngine {
    pub emotional_space: VectorSpace,
    pub mood_chain: MarkovChain,
    pub emotional_weights: Array1<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySystem {
    pub short_term_memory: VecDeque<MemoryEntry>,
    pub long_term_memory: Vec<MemoryEntry>,
    pub emotional_significance: HashMap<String, f32>,  // Memory ID to emotional significance
    pub memory_consolidation: HashMap<String, f32>,    // Memory ID to consolidation strength
    pub context_links: HashMap<String, Vec<String>>,   // Context to memory IDs
    pub temporal_links: HashMap<String, Vec<String>>,  // Time-based memory connections
    pub semantic_links: HashMap<String, Vec<String>>,  // Concept-based memory connections
    pub working_memory: Vec<MemoryEntry>,              // Currently active memories
    pub memory_decay_rate: f32,
    pub consolidation_threshold: f32,
    pub max_short_term_capacity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub context: String,
    pub emotional_vector: Array1<f32>,
    pub semantic_vector: Array1<f32>,
    pub importance: f32,
    pub access_count: u32,
    pub last_accessed: DateTime<Utc>,
    pub related_memories: Vec<String>,
    pub memory_type: MemoryType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    Conversation,
    EmotionalExperience,
    Fact,
    Reflection,
    Goal,
    Decision,
}

impl MemorySystem {
    pub fn new(dimensions: usize) -> Self {
        Self {
            short_term_memory: VecDeque::with_capacity(100),
            long_term_memory: Vec::new(),
            emotional_significance: HashMap::new(),
            memory_consolidation: HashMap::new(),
            context_links: HashMap::new(),
            temporal_links: HashMap::new(),
            semantic_links: HashMap::new(),
            working_memory: Vec::with_capacity(7),  // Miller's magic number
            memory_decay_rate: 0.1,
            consolidation_threshold: 0.7,
            max_short_term_capacity: 100,
        }
    }

    pub fn store_memory(&mut self, content: String, context: String, 
                       emotional_vector: Array1<f32>, semantic_vector: Array1<f32>,
                       memory_type: MemoryType) -> Result<()> {
        let id = format!("memory_{}", Utc::now().timestamp());
        let importance = self.calculate_importance(&emotional_vector, &semantic_vector)?;
        
        let entry = MemoryEntry {
            id: id.clone(),
            content: content.clone(),
            timestamp: Utc::now(),
            context: context.clone(),
            emotional_vector,
            semantic_vector,
            importance,
            access_count: 1,
            last_accessed: Utc::now(),
            related_memories: Vec::new(),
            memory_type,
        };

        // Store in short-term memory
        if self.short_term_memory.len() >= self.max_short_term_capacity {
            self.short_term_memory.pop_front();
        }
        self.short_term_memory.push_back(entry.clone());

        // Update context links
        self.context_links.entry(context.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        // Update emotional significance
        self.emotional_significance.insert(id.clone(), importance);

        // Check for consolidation
        if importance >= self.consolidation_threshold {
            self.consolidate_memory(&entry)?;
        }

        Ok(())
    }

    fn calculate_importance(&self, emotional_vector: &Array1<f32>, semantic_vector: &Array1<f32>) -> Result<f32> {
        // Calculate importance based on emotional intensity and semantic relevance
        let emotional_intensity = emotional_vector.dot(emotional_vector).sqrt();
        let semantic_relevance = semantic_vector.dot(semantic_vector).sqrt();
        
        Ok((emotional_intensity * 0.6 + semantic_relevance * 0.4).max(0.0).min(1.0))
    }

    fn consolidate_memory(&mut self, entry: &MemoryEntry) -> Result<()> {
        // Move to long-term memory
        self.long_term_memory.push(entry.clone());
        
        // Initialize consolidation strength
        self.memory_consolidation.insert(entry.id.clone(), 1.0);
        
        // Find related memories
        self.find_related_memories(entry)?;
        
        Ok(())
    }

    fn find_related_memories(&mut self, entry: &MemoryEntry) -> Result<()> {
        let mut related = Vec::new();
        
        // Find temporal relations
        if let Some(temporal_memories) = self.temporal_links.get(&entry.context) {
            for memory_id in temporal_memories {
                if self.is_related_temporally(entry, memory_id)? {
                    related.push(memory_id.clone());
                }
            }
        }
        
        // Find semantic relations
        for (other_id, other_entry) in self.get_all_memories() {
            if self.is_related_semantically(&entry.semantic_vector, &other_entry.semantic_vector)? {
                related.push(other_id);
            }
        }
        
        // Update related memories
        if let Some(entry) = self.get_memory_mut(&entry.id) {
            entry.related_memories = related;
        }
        
        Ok(())
    }

    pub fn retrieve_memory(&mut self, query: &str, context: Option<&str>) -> Result<Vec<MemoryEntry>> {
        let mut results = Vec::new();
        
        // Search in working memory first
        for entry in &self.working_memory {
            if entry.content.contains(query) {
                results.push(entry.clone());
            }
        }
        
        // Search in short-term memory
        for entry in &self.short_term_memory {
            if entry.content.contains(query) {
                results.push(entry.clone());
            }
        }
        
        // Search in long-term memory
        for entry in &self.long_term_memory {
            if entry.content.contains(query) {
                results.push(entry.clone());
            }
        }
        
        // Filter by context if provided
        if let Some(ctx) = context {
            results.retain(|entry| entry.context == ctx);
        }
        
        // Sort by relevance
        results.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        
        Ok(results)
    }

    pub fn reflect_on_memory(&self, memory_id: &str) -> Result<String> {
        let entry = self.get_memory(memory_id)?;
        
        // Generate reflection based on memory type and content
        let reflection = match entry.memory_type {
            MemoryType::Conversation => self.reflect_on_conversation(&entry)?,
            MemoryType::EmotionalExperience => self.reflect_on_emotion(&entry)?,
            MemoryType::Decision => self.reflect_on_decision(&entry)?,
            _ => format!("Reflecting on: {}", entry.content),
        };
        
        Ok(reflection)
    }

    fn reflect_on_conversation(&self, entry: &MemoryEntry) -> Result<String> {
        let mut reflection = String::new();
        
        // Analyze emotional context
        let emotional_state = self.analyze_emotional_state(&entry.emotional_vector)?;
        reflection.push_str(&format!("I felt {} during this conversation. ", emotional_state));
        
        // Check for related memories
        if !entry.related_memories.is_empty() {
            reflection.push_str("This reminds me of previous conversations where ");
            for related_id in &entry.related_memories {
                if let Ok(related) = self.get_memory(related_id) {
                    reflection.push_str(&format!("{} ", related.content));
                }
            }
        }
        
        Ok(reflection)
    }

    fn reflect_on_emotion(&self, entry: &MemoryEntry) -> Result<String> {
        let emotional_state = self.analyze_emotional_state(&entry.emotional_vector)?;
        let importance = self.emotional_significance.get(&entry.id).unwrap_or(&0.0);
        
        Ok(format!(
            "I experienced strong {} feelings (significance: {:.2}). This emotional memory is {}.",
            emotional_state,
            importance,
            if *importance > 0.7 { "deeply ingrained" } else { "still forming" }
        ))
    }

    fn analyze_emotional_state(&self, vector: &Array1<f32>) -> Result<String> {
        // Simple emotional state analysis based on vector components
        let positive = vector.iter().filter(|&&x| x > 0.0).count() as f32 / vector.len() as f32;
        let intensity = vector.dot(vector).sqrt();
        
        Ok(format!(
            "{} and {}",
            if positive > 0.6 { "positive" } else if positive < 0.4 { "negative" } else { "mixed" },
            if intensity > 0.7 { "intense" } else if intensity > 0.3 { "moderate" } else { "mild" }
        ))
    }

    pub fn update_working_memory(&mut self, memory_id: &str) -> Result<()> {
        if let Some(entry) = self.get_memory(memory_id) {
            // Remove oldest entry if at capacity
            if self.working_memory.len() >= 7 {
                self.working_memory.remove(0);
            }
            
            // Add to working memory
            self.working_memory.push(entry.clone());
            
            // Update access count and timestamp
            if let Some(entry) = self.get_memory_mut(memory_id) {
                entry.access_count += 1;
                entry.last_accessed = Utc::now();
            }
        }
        
        Ok(())
    }

    pub fn get_memory(&self, memory_id: &str) -> Result<MemoryEntry> {
        // Search in working memory
        if let Some(entry) = self.working_memory.iter().find(|e| e.id == memory_id) {
            return Ok(entry.clone());
        }
        
        // Search in short-term memory
        if let Some(entry) = self.short_term_memory.iter().find(|e| e.id == memory_id) {
            return Ok(entry.clone());
        }
        
        // Search in long-term memory
        if let Some(entry) = self.long_term_memory.iter().find(|e| e.id == memory_id) {
            return Ok(entry.clone());
        }
        
        Err(MathEngineError::InvalidParameter("Memory not found".into()))
    }

    fn get_memory_mut(&mut self, memory_id: &str) -> Option<&mut MemoryEntry> {
        // Search in working memory
        if let Some(entry) = self.working_memory.iter_mut().find(|e| e.id == memory_id) {
            return Some(entry);
        }
        
        // Search in short-term memory
        if let Some(entry) = self.short_term_memory.iter_mut().find(|e| e.id == memory_id) {
            return Some(entry);
        }
        
        // Search in long-term memory
        if let Some(entry) = self.long_term_memory.iter_mut().find(|e| e.id == memory_id) {
            return Some(entry);
        }
        
        None
    }

    fn get_all_memories(&self) -> Vec<(String, MemoryEntry)> {
        let mut all_memories = Vec::new();
        
        // Add working memory
        for entry in &self.working_memory {
            all_memories.push((entry.id.clone(), entry.clone()));
        }
        
        // Add short-term memory
        for entry in &self.short_term_memory {
            all_memories.push((entry.id.clone(), entry.clone()));
        }
        
        // Add long-term memory
        for entry in &self.long_term_memory {
            all_memories.push((entry.id.clone(), entry.clone()));
        }
        
        all_memories
    }

    pub fn process_decay(&mut self) -> Result<()> {
        // Process decay for short-term memory
        let mut to_remove = Vec::new();
        for (i, entry) in self.short_term_memory.iter().enumerate() {
            let time_since_access = (Utc::now() - entry.last_accessed).num_seconds() as f32;
            let decay = self.memory_decay_rate * time_since_access;
            
            if let Some(importance) = self.emotional_significance.get_mut(&entry.id) {
                *importance = (*importance - decay).max(0.0);
                
                if *importance < 0.1 {
                    to_remove.push(i);
                }
            }
        }
        
        // Remove decayed memories
        for &i in to_remove.iter().rev() {
            if let Some(entry) = self.short_term_memory.remove(i) {
                self.emotional_significance.remove(&entry.id);
                self.memory_consolidation.remove(&entry.id);
            }
        }
        
        Ok(())
    }

    pub fn integrate_emotional_memory(&mut self, memory_id: &str, emotional_context: &Array1<f32>) -> Result<()> {
        let entry = self.get_memory_mut(memory_id)
            .ok_or_else(|| MathEngineError::InvalidParameter("Memory not found".into()))?;

        // Calculate emotional significance
        let emotional_significance = emotional_context.dot(emotional_context).sqrt();
        
        // Update memory with emotional context
        entry.emotional_vector = &entry.emotional_vector * 0.7 + emotional_context * 0.3;
        entry.importance = (entry.importance * 0.6 + emotional_significance * 0.4).max(0.0).min(1.0);

        // Update consolidation strength
        let consolidation = self.memory_consolidation.entry(memory_id.to_string())
            .or_insert(0.0);
        *consolidation = (*consolidation * 0.8 + emotional_significance * 0.2).min(1.0);

        // Find and update related memories
        self.update_related_memories(memory_id, emotional_context)?;

        Ok(())
    }

    fn update_related_memories(&mut self, memory_id: &str, emotional_context: &Array1<f32>) -> Result<()> {
        let entry = self.get_memory(memory_id)?;
        
        // Find memories with similar emotional context
        let mut related_memories = Vec::new();
        for (other_id, other_entry) in self.get_all_memories() {
            if other_id != memory_id {
                let emotional_similarity = entry.emotional_vector.dot(&other_entry.emotional_vector);
                let semantic_similarity = entry.semantic_vector.dot(&other_entry.semantic_vector);
                
                if emotional_similarity > 0.7 || semantic_similarity > 0.7 {
                    related_memories.push((other_id, emotional_similarity, semantic_similarity));
                }
            }
        }

        // Update related memories
        for (related_id, emotional_sim, semantic_sim) in related_memories {
            if let Some(related_entry) = self.get_memory_mut(&related_id) {
                // Strengthen emotional connection
                related_entry.emotional_vector = &related_entry.emotional_vector * 0.9 + 
                                              emotional_context * 0.1 * emotional_sim;
                
                // Update importance based on connection strength
                let connection_strength = (emotional_sim + semantic_sim) / 2.0;
                related_entry.importance = (related_entry.importance * 0.8 + 
                                          connection_strength * 0.2).max(0.0).min(1.0);
                
                // Add to related memories if not already present
                if !related_entry.related_memories.contains(&memory_id.to_string()) {
                    related_entry.related_memories.push(memory_id.to_string());
                }
            }
        }

        Ok(())
    }

    pub fn consolidate_related_memories(&mut self, memory_id: &str) -> Result<()> {
        let entry = self.get_memory(memory_id)?;
        
        // Get all related memories
        let mut related_entries = Vec::new();
        for related_id in &entry.related_memories {
            if let Ok(related_entry) = self.get_memory(related_id) {
                related_entries.push(related_entry);
            }
        }

        if related_entries.is_empty() {
            return Ok(());
        }

        // Calculate consolidated emotional and semantic vectors
        let mut consolidated_emotional = Array1::zeros(entry.emotional_vector.len());
        let mut consolidated_semantic = Array1::zeros(entry.semantic_vector.len());
        let mut total_weight = 0.0;

        for related_entry in &related_entries {
            let weight = related_entry.importance * 
                        self.memory_consolidation.get(&related_entry.id).unwrap_or(&0.0);
            
            consolidated_emotional = consolidated_emotional + &related_entry.emotional_vector * weight;
            consolidated_semantic = consolidated_semantic + &related_entry.semantic_vector * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            consolidated_emotional = consolidated_emotional / total_weight;
            consolidated_semantic = consolidated_semantic / total_weight;
        }

        // Update memory with consolidated information
        if let Some(entry) = self.get_memory_mut(memory_id) {
            entry.emotional_vector = &entry.emotional_vector * 0.6 + &consolidated_emotional * 0.4;
            entry.semantic_vector = &entry.semantic_vector * 0.6 + &consolidated_semantic * 0.4;
            
            // Update consolidation strength
            let consolidation = self.memory_consolidation.entry(memory_id.to_string())
                .or_insert(0.0);
            *consolidation = (*consolidation * 0.7 + total_weight * 0.3).min(1.0);
        }

        Ok(())
    }

    pub fn retrieve_contextual_memories(&self, context: &str, emotional_state: &Array1<f32>) -> Result<Vec<MemoryEntry>> {
        let mut relevant_memories = Vec::new();
        
        // Get all memories in the context
        if let Some(memory_ids) = self.context_links.get(context) {
            for memory_id in memory_ids {
                if let Ok(entry) = self.get_memory(memory_id) {
                    // Calculate relevance score
                    let emotional_relevance = entry.emotional_vector.dot(emotional_state);
                    let time_relevance = 1.0 - (Utc::now() - entry.last_accessed).num_seconds() as f32 / 
                                       (24.0 * 3600.0); // Normalize to 24 hours
                    
                    let relevance = (emotional_relevance * 0.6 + time_relevance * 0.4)
                        .max(0.0).min(1.0);
                    
                    if relevance > 0.5 { // Threshold for relevance
                        relevant_memories.push((entry, relevance));
                    }
                }
            }
        }

        // Sort by relevance
        relevant_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(relevant_memories.into_iter().map(|(entry, _)| entry).collect())
    }

    pub fn update_memory_consolidation(&mut self) -> Result<()> {
        let mut to_consolidate = Vec::new();
        
        // Find memories ready for consolidation
        for (memory_id, consolidation) in &self.memory_consolidation {
            if *consolidation > this.consolidation_threshold {
                to_consolidate.push(memory_id.clone());
            }
        }

        // Consolidate memories
        for memory_id in to_consolidate {
            this.consolidate_related_memories(&memory_id)?;
        }

        Ok(())
    }
}

// Integration layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalitySystem {
    pub trait_engine: TraitEngine,
    pub mood_engine: MoodEngine,
    pub memory_engine: MemorySystem,
    pub integration_matrix: Array2<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub trait_stability: f32,
    pub mood_entropy: f32,
    pub memory_entropy: f32,
    pub overall_stability: f32,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl PersonalitySystem {
    pub fn new(dimensions: usize) -> Result<Self> {
        if dimensions == 0 {
            return Err(MathEngineError::InvalidParameter("Dimensions must be positive".into()));
        }

        // Initialize integration matrix with proper normalization and stability checks
        let mut integration_matrix = Array2::from_shape_fn((dimensions, dimensions), |_| rand::random::<f32>());
        let norm = integration_matrix.dot(&integration_matrix.t()).diag().sum().sqrt();
        
        if norm < 1e-10 {
            return Err(MathEngineError::NumericalInstability(
                "Integration matrix normalization failed".into()
            ));
        }
        
        integration_matrix.mapv_inplace(|x| x / norm);

        Ok(Self {
            trait_engine: TraitEngine::new(dimensions)?,
            mood_engine: MoodEngine::new(dimensions),
            memory_engine: MemorySystem::new(dimensions),
            integration_matrix,
        })
    }

    pub fn process_interaction(&mut self, trait_id: &str, stimulus: Array1<f32>) -> Result<SystemState> {
        // Validate input dimensions
        if stimulus.len() != self.trait_engine.trait_space.dimensions {
            return Err(MathEngineError::DimensionMismatch {
                expected: self.trait_engine.trait_space.dimensions,
                got: stimulus.len(),
            });
        }

        // Update trait with error handling
        self.trait_engine.update_trait(trait_id, stimulus.clone(), "")?;
        
        // Get the updated trait vector
        let trait_vector = self.trait_engine.trait_space.vectors.get(trait_id)
            .ok_or_else(|| MathEngineError::InvalidParameter("Trait not found".into()))?;

        // Decompose the trait vector into eigenbasis
        let eigen_components = self.decompose_into_eigenbasis(trait_vector)?;
        
        // Identify dominant components (those with magnitude > threshold)
        let dominant_components = self.identify_dominant_components(&eigen_components)?;
        
        // Reconstruct emotional context using only dominant components
        let emotional_context = self.reconstruct_emotional_context(&dominant_components)?;
        
        // Update mood using the reconstructed emotional context
        self.mood_engine.update_mood(&emotional_context)?;
        
        // Store memory of interaction with component information
        let memory_id = format!("interaction_{}", trait_id);
        let memory_vector = self.create_memory_vector(&stimulus, &eigen_components)?;
        self.memory_engine.store_memory(&memory_id, memory_vector)?;

        // Calculate and return system state
        self.calculate_system_state()
    }

    fn decompose_into_eigenbasis(&self, vector: &Array1<f32>) -> Result<Array1<f32>> {
        // Project vector onto eigenbasis
        let eigenbasis = &this.trait_engine.interaction_matrix.eigenvectors;
        let components = eigenbasis.t().dot(vector);
        
        // Validate decomposition
        if components.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(MathEngineError::NumericalInstability(
                "Eigen decomposition produced invalid values".into()
            ));
        }
        
        Ok(components)
    }

    fn identify_dominant_components(&self, components: &Array1<f32>) -> Result<Vec<(usize, f32)>> {
        // Calculate component magnitudes
        let magnitudes: Vec<(usize, f32)> = components.iter()
            .enumerate()
            .map(|(i, &x)| (i, x.abs()))
            .collect();
        
        // Sort by magnitude
        let mut sorted_magnitudes = magnitudes.clone();
        sorted_magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Calculate total energy
        let total_energy: f32 = magnitudes.iter().map(|(_, m)| m * m).sum();
        
        // Find components that contribute to 95% of total energy
        let mut cumulative_energy = 0.0;
        let mut dominant_components = Vec::new();
        
        for (i, magnitude) in sorted_magnitudes {
            cumulative_energy += magnitude * magnitude;
            dominant_components.push((i, components[i]));
            
            if cumulative_energy / total_energy >= 0.95 {
                break;
            }
        }
        
        Ok(dominant_components)
    }

    fn reconstruct_emotional_context(&self, dominant_components: &[(usize, f32)]) -> Result<Array1<f32>> {
        // Initialize emotional context vector
        let mut emotional_context = Array1::zeros(this.trait_engine.trait_space.dimensions);
        
        // Reconstruct using dominant components
        let eigenbasis = &this.trait_engine.interaction_matrix.eigenvectors;
        
        for &(i, value) in dominant_components {
            let eigenvector = eigenbasis.column(i);
            emotional_context = emotional_context + eigenvector * value;
        }
        
        // Normalize the reconstructed context
        let norm = emotional_context.dot(&emotional_context).sqrt();
        if norm < 1e-10 {
            return Err(MathEngineError::NumericalInstability(
                "Reconstructed emotional context has zero magnitude".into()
            ));
        }
        
        Ok(emotional_context.mapv(|x| x / norm))
    }

    fn create_memory_vector(&self, stimulus: &Array1<f32>, components: &Array1<f32>) -> Result<Array1<f32>> {
        // Create a memory vector that includes both raw stimulus and component information
        let mut memory_vector = Array1::zeros(stimulus.len() * 2);
        
        // First half contains the original stimulus
        for i in 0..stimulus.len() {
            memory_vector[i] = stimulus[i];
        }
        
        // Second half contains the component magnitudes
        for i in 0..components.len() {
            memory_vector[stimulus.len() + i] = components[i].abs();
        }
        
        // Normalize the memory vector
        let norm = memory_vector.dot(&memory_vector).sqrt();
        if norm < 1e-10 {
            return Err(MathEngineError::NumericalInstability(
                "Memory vector has zero magnitude".into()
            ));
        }
        
        Ok(memory_vector.mapv(|x| x / norm))
    }

    fn calculate_system_state(&self) -> Result<SystemState> {
        let trait_stability = this.trait_engine.interaction_matrix.calculate_stability()?;
        let mood_entropy = this.mood_engine.mood_chain.calculate_entropy();
        let memory_entropy = this.memory_engine.calculate_memory_entropy();
        
        // Validate metrics
        if !trait_stability.is_finite() || !mood_entropy.is_finite() || !memory_entropy.is_finite() {
            return Err(MathEngineError::NumericalInstability(
                "Invalid system metrics detected".into()
            ));
        }

        let overall_stability = 0.4 * trait_stability + 
                              0.3 * (1.0 - mood_entropy) + 
                              0.3 * (1.0 - memory_entropy);

        Ok(SystemState {
            trait_stability,
            mood_entropy,
            memory_entropy,
            overall_stability,
            last_update: chrono::Utc::now(),
        })
    }

    pub fn get_system_metrics(&self) -> HashMap<String, f32> {
        let state = this.calculate_system_state().unwrap();
        let mut metrics = HashMap::new();
        
        metrics.insert("trait_stability".to_string(), state.trait_stability);
        metrics.insert("mood_entropy".to_string(), state.mood_entropy);
        metrics.insert("memory_entropy".to_string(), state.memory_entropy);
        metrics.insert("overall_stability".to_string(), state.overall_stability);
        
        // Add component-specific metrics
        if let Some(trait_vector) = this.trait_engine.trait_space.vectors.get("current") {
            metrics.insert("trait_vector_norm".to_string(), trait_vector.dot(trait_vector).sqrt());
        }
        
        if let Some(mood_vector) = this.mood_engine.emotional_space.vectors.get("current_mood") {
            metrics.insert("mood_vector_norm".to_string(), mood_vector.dot(mood_vector).sqrt());
        }
        
        metrics.insert("memory_count".to_string(), this.memory_engine.memory_space.vectors.len() as f32);
        
        metrics
    }

    pub fn visualize_state(&self) -> String {
        let metrics = this.get_system_metrics();
        let mut visualization = String::new();
        
        visualization.push_str("System State Visualization:\n");
        visualization.push_str("==========================\n");
        
        for (key, value) in metrics {
            visualization.push_str(&format!("{}: {:.3}\n", key, value));
        }
        
        visualization.push_str("\nTrait Space:\n");
        if let Some(trait_vector) = this.trait_engine.trait_space.vectors.get("current") {
            visualization.push_str(&format!("Current trait vector: {:?}\n", trait_vector));
        }
        
        visualization.push_str("\nMood Space:\n");
        if let Some(mood_vector) = this.mood_engine.emotional_space.vectors.get("current_mood") {
            visualization.push_str(&format!("Current mood vector: {:?}\n", mood_vector));
        }
        
        visualization
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalIntelligence {
    pub empathy_vector: Array1<f32>,
    pub emotional_predictions: HashMap<String, Array1<f32>>,
    pub social_emotional_patterns: HashMap<String, Vec<Array1<f32>>>,
    pub emotional_contagion: f32,
    pub empathy_strength: f32,
    pub emotional_memory: VecDeque<(DateTime<Utc>, String, Array1<f32>, f32)>, // (timestamp, context, emotion, intensity)
    pub emotional_resonance: HashMap<String, f32>,  // Resonance with different people
    pub social_contexts: HashMap<String, Array1<f32>>,  // Emotional patterns in different social contexts
    pub emotional_adaptation: f32,  // Ability to adapt emotional responses
    pub emotional_depth: HashMap<String, f32>,  // Depth of understanding for different emotions
    pub emotional_insights: VecDeque<(DateTime<Utc>, String, f32)>, // (timestamp, insight, confidence)
    pub emotional_patterns: HashMap<String, EmotionalPattern>,
    pub emotional_hysteresis: HashMap<String, Array1<f32>>,
    pub emotional_memory_weights: Array1<f32>, // Weights for valence, arousal, dominance
}

#[derive(Debug, Clone)]
pub struct EmotionalPattern {
    pub frequency: usize,
    pub average_emotion: Array1<f32>,
    pub variance: Array1<f32>,
    pub last_observed: DateTime<Utc>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStateTransition {
    pub from_state: Array1<f32>,
    pub to_state: Array1<f32>,
    pub transition_probability: f32,
    pub transition_time: Duration,
    pub context: String,
    pub triggers: Vec<String>,
    pub success_rate: f32,
    pub last_observed: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalRegulationStrategy {
    pub strategy_id: String,
    pub target_state: Array1<f32>,
    pub regulation_steps: Vec<Array1<f32>>,
    pub success_rate: f32,
    pub average_time: Duration,
    pub context_specificity: f32,
    pub energy_cost: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalPatternModel {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub learning_rate: f32,
    pub epochs: usize,
    pub accuracy: f32,
    pub last_trained: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStateVisualization {
    pub state_vector: Array1<f32>,
    pub dominant_emotions: Vec<(String, f32)>,
    pub intensity_map: Vec<f32>,
    pub transition_path: Vec<Array1<f32>>,
    pub stability_score: f32,
    pub context_influence: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStateAnalysis {
    pub primary_emotions: Vec<(String, f32)>,
    pub secondary_emotions: Vec<(String, f32)>,
    pub emotional_intensity: f32,
    pub emotional_stability: f32,
    pub emotional_complexity: f32,
    pub emotional_ambivalence: f32,
    pub emotional_resonance: HashMap<String, f32>,
    pub emotional_context: HashMap<String, f32>,
    pub emotional_history: Vec<(DateTime<Utc>, Array1<f32>)>,
    pub emotional_trends: Vec<(String, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalComponent {
    pub name: String,
    pub intensity: f32,
    pub variance: f32,
    pub persistence: f32,
    pub influence: f32,
    pub related_components: Vec<String>,
}

impl EmotionalIntelligence {
    pub fn new(dimensions: usize) -> Self {
        Self {
            empathy_vector: Array1::ones(dimensions) / dimensions as f32,
            emotional_predictions: HashMap::new(),
            social_emotional_patterns: HashMap::new(),
            emotional_contagion: 0.5,
            empathy_strength: 0.7,
            emotional_memory: VecDeque::with_capacity(1000),
            emotional_resonance: HashMap::new(),
            social_contexts: HashMap::new(),
            emotional_adaptation: 0.5,
            emotional_depth: HashMap::new(),
            emotional_insights: VecDeque::with_capacity(100),
            emotional_patterns: HashMap::new(),
            emotional_hysteresis: HashMap::new(),
            emotional_memory_weights: Array1::ones(3) / 3.0, // Weights for valence, arousal, dominance
        }
    }

    pub fn analyze_emotional_pattern(&mut self, person_id: &str, context: &str, 
                                   emotion: &Array1<f32>) -> Result<()> {
        // Calculate pattern key
        let pattern_key = format!("{}:{}", person_id, context);
        
        // Get or create pattern
        let pattern = this.emotional_patterns.entry(pattern_key.clone())
            .or_insert_with(|| EmotionalPattern {
                frequency: 0,
                average_emotion: emotion.clone(),
                variance: Array1::zeros(emotion.len()),
                last_observed: Utc::now(),
                confidence: 0.0,
            });
        
        // Update pattern statistics
        pattern.frequency += 1;
        let n = pattern.frequency as f32;
        
        // Update average using Welford's algorithm
        let delta = emotion - &pattern.average_emotion;
        pattern.average_emotion = &pattern.average_emotion + &(&delta / n);
        
        // Update variance
        if n > 1.0 {
            pattern.variance = &pattern.variance + &(&delta * &(emotion - &pattern.average_emotion));
        }
        
        // Update confidence based on frequency and variance
        let variance_norm = pattern.variance.dot(&pattern.variance).sqrt();
        pattern.confidence = (1.0 - variance_norm).max(0.0) * (1.0 - (-n / 10.0).exp());
        
        // Update last observed time
        pattern.last_observed = Utc::now();
        
        Ok(())
    }

    pub fn get_emotional_prediction(&self, person_id: &str, context: &str) -> Option<Array1<f32>> {
        let pattern_key = format!("{}:{}", person_id, context);
        this.emotional_patterns.get(&pattern_key)
            .map(|pattern| pattern.average_emotion.clone())
    }

    pub fn update_emotional_hysteresis(&mut self, person_id: &str, emotion: &Array1<f32>) {
        let hysteresis = this.emotional_hysteresis.entry(person_id.to_string())
            .or_insert_with(|| Array1::zeros(emotion.len()));
        
        // Apply hysteresis effect (memory of past emotional states)
        *hysteresis = &*hysteresis * 0.9 + emotion * 0.1;
    }

    pub fn get_emotional_memory_weight(&self, memory: &EmotionalAnalysis) -> f32 {
        // Calculate weighted importance of emotional memory
        let weights = &this.emotional_memory_weights;
        memory.valence * weights[0] + 
        memory.arousal * weights[1] + 
        memory.dominance * weights[2]
    }

    pub fn process_emotional_state(&mut self, person_id: &str, context: &str, 
                                 emotion: &Array1<f32>) -> Result<()> {
        // Calculate emotional intensity
        let intensity = emotion.dot(emotion).sqrt();
        
        // Store emotional memory
        if this.emotional_memory.len() >= 1000 {
            this.emotional_memory.pop_front();
        }
        this.emotional_memory.push_back((
            Utc::now(),
            context.to_string(),
            emotion.clone(),
            intensity
        ));

        // Update emotional patterns
        this.analyze_emotional_pattern(person_id, context, emotion)?;
        
        // Update emotional hysteresis
        this.update_emotional_hysteresis(person_id, emotion);
        
        // Update social context understanding
        this.update_social_context(context, emotion)?;
        
        // Generate emotional insights
        this.generate_emotional_insights(person_id, context, emotion)?;
        
        // Update emotional adaptation
        this.update_emotional_adaptation(emotion)?;

        Ok(())
    }

    pub fn analyze_emotional_trends(&self, person_id: &str, time_window: Duration) -> Result<EmotionalTrend> {
        let cutoff = Utc::now() - time_window;
        let relevant_memories: Vec<_> = this.emotional_memory
            .iter()
            .filter(|(time, id, _, _)| *time >= cutoff && id == person_id)
            .collect();

        if relevant_memories.is_empty() {
            return Err(MathEngineError::InvalidParameter("No emotional data in time window".into()));
        }

        // Calculate trend statistics
        let mut trend_vector = Array1::zeros(relevant_memories[0].2.len());
        let mut intensity_sum = 0.0;
        let mut time_weighted_sum = 0.0;
        let mut count = 0;

        for (time, _, emotion, intensity) in &relevant_memories {
            let time_weight = 1.0 - (Utc::now() - *time).num_seconds() as f32 / time_window.num_seconds() as f32;
            trend_vector = trend_vector + emotion * time_weight * intensity;
            intensity_sum += intensity;
            time_weighted_sum += time_weight;
            count += 1;
        }

        // Normalize trend vector
        if count > 0 {
            trend_vector = trend_vector / (intensity_sum * time_weighted_sum);
        }

        // Calculate volatility
        let mut volatility = 0.0;
        for (_, _, emotion, intensity) in &relevant_memories {
            let diff = emotion - &trend_vector;
            volatility += diff.dot(&diff) * intensity;
        }
        volatility = (volatility / count as f32).sqrt();

        Ok(EmotionalTrend {
            trend_vector,
            volatility,
            intensity: intensity_sum / count as f32,
            sample_count: count,
        })
    }

    pub fn predict_emotional_response(&self, person_id: &str, context: &str, 
                                    stimulus: &Array1<f32>) -> Result<EmotionalPrediction> {
        // Get baseline emotional state
        let baseline = this.get_emotional_prediction(person_id, context)
            .ok_or_else(|| MathEngineError::InvalidParameter("No baseline emotional data".into()))?;

        // Get emotional pattern
        let pattern_key = format!("{}:{}", person_id, context);
        let pattern = this.emotional_patterns.get(&pattern_key)
            .ok_or_else(|| MathEngineError::InvalidParameter("No emotional pattern found".into()))?;

        // Calculate stimulus impact
        let stimulus_impact = stimulus.dot(&this.empathy_vector) * this.empathy_strength;

        // Calculate predicted emotion
        let mut predicted_emotion = &baseline + &(stimulus * stimulus_impact);

        // Apply emotional hysteresis
        if let Some(hysteresis) = this.emotional_hysteresis.get(person_id) {
            predicted_emotion = &predicted_emotion * 0.7 + hysteresis * 0.3;
        }

        // Calculate confidence
        let confidence = pattern.confidence * (1.0 - pattern.variance.dot(&pattern.variance).sqrt());

        Ok(EmotionalPrediction {
            predicted_emotion,
            confidence,
            baseline_emotion: baseline,
            stimulus_impact,
        })
    }

    pub fn analyze_emotional_correlation(&self, person_id: &str, context: &str) -> Result<EmotionalCorrelation> {
        let pattern_key = format!("{}:{}", person_id, context);
        let pattern = this.emotional_patterns.get(&pattern_key)
            .ok_or_else(|| MathEngineError::InvalidParameter("No emotional pattern found".into()))?;

        // Calculate correlation matrix
        let mut correlation_matrix = Array2::zeros((pattern.average_emotion.len(), pattern.average_emotion.len()));
        
        for i in 0..pattern.average_emotion.len() {
            for j in 0..pattern.average_emotion.len() {
                let cov = pattern.variance[i] * pattern.variance[j];
                correlation_matrix[[i, j]] = if cov > 0.0 {
                    pattern.average_emotion[i] * pattern.average_emotion[j] / cov
                } else {
                    0.0
                };
            }
        }

        // Calculate dominant emotional components
        let (eigenvalues, eigenvectors) = correlation_matrix.eig()
            .map_err(|e| MathEngineError::DecompositionError(e.to_string()))?;

        let mut dominant_components = Vec::new();
        for (i, &value) in eigenvalues.iter().enumerate() {
            if value.abs() > 0.5 { // Threshold for significant components
                dominant_components.push((i, value, eigenvectors.column(i).to_owned()));
            }
        }

        Ok(EmotionalCorrelation {
            correlation_matrix,
            dominant_components,
            pattern_strength: pattern.confidence,
        })
    }

    pub fn regulate_emotion(&mut self, person_id: &str, current_emotion: &Array1<f32>, 
                          target_emotion: &Array1<f32>) -> Result<Array1<f32>> {
        // Calculate emotional distance
        let distance = (current_emotion - target_emotion).dot(&(current_emotion - target_emotion)).sqrt();
        
        // Get emotional hysteresis
        let hysteresis = this.emotional_hysteresis.get(person_id)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(current_emotion.len()));

        // Calculate regulation strength based on distance and adaptation
        let regulation_strength = (distance * this.emotional_adaptation).min(1.0);

        // Apply regulation with hysteresis consideration
        let regulated_emotion = current_emotion * (1.0 - regulation_strength) +
                              target_emotion * regulation_strength * 0.7 +
                              &hysteresis * regulation_strength * 0.3;

        // Update emotional hysteresis
        this.update_emotional_hysteresis(person_id, &regulated_emotion);

        // Store regulation attempt
        this.emotional_memory.push_back((
            Utc::now(),
            person_id.to_string(),
            regulated_emotion.clone(),
            regulation_strength
        ));

        Ok(regulated_emotion)
    }

    pub fn analyze_state_transitions(&mut self, person_id: &str) -> Result<Vec<EmotionalStateTransition>> {
        let mut transitions = Vec::new();
        let memories: Vec<_> = this.emotional_memory.iter().collect();
        
        for window in memories.windows(2) {
            let (time1, _, state1, _) = window[0];
            let (time2, _, state2, _) = window[1];
            
            let transition = EmotionalStateTransition {
                from_state: state1.clone(),
                to_state: state2.clone(),
                transition_probability: 1.0,
                transition_time: *time2 - *time1,
                context: "".to_string(),
                triggers: Vec::new(),
                success_rate: 1.0,
                last_observed: *time2,
            };
            
            transitions.push(transition);
        }
        
        Ok(transitions)
    }

    pub fn develop_regulation_strategy(&mut self, current_state: &Array1<f32>, 
                                     target_state: &Array1<f32>) -> Result<EmotionalRegulationStrategy> {
        // Calculate direct path
        let direct_path = target_state - current_state;
        let distance = direct_path.dot(&direct_path).sqrt();
        
        // Generate intermediate steps
        let num_steps = (distance * 10.0).ceil() as usize;
        let mut steps = Vec::with_capacity(num_steps);
        
        for i in 0..num_steps {
            let progress = (i as f32 + 1.0) / (num_steps as f32 + 1.0);
            let step = current_state + &(&direct_path * progress);
            steps.push(step);
        }
        
        // Calculate strategy metrics
        let energy_cost = distance * 0.5; // Base energy cost
        let context_specificity = 0.7; // Default specificity
        
        Ok(EmotionalRegulationStrategy {
            strategy_id: format!("reg_{}", Utc::now().timestamp()),
            target_state: target_state.clone(),
            regulation_steps: steps,
            success_rate: 0.8, // Initial estimate
            average_time: Duration::seconds((num_steps * 2) as i64),
            context_specificity,
            energy_cost,
        })
    }

    pub fn apply_regulation_strategy(&mut self, person_id: &str, 
                                   strategy: &EmotionalRegulationStrategy) -> Result<Array1<f32>> {
        let mut current_state = this.get_current_emotional_state(person_id)?;
        let mut success_count = 0;
        let mut total_steps = 0;
        
        for step in &strategy.regulation_steps {
            // Calculate regulation strength
            let distance = (step - &current_state).dot(&(step - &current_state)).sqrt();
            let regulation_strength = (distance * this.emotional_adaptation).min(1.0);
            
            // Apply regulation with hysteresis
            let hysteresis = this.emotional_hysteresis.get(person_id)
                .cloned()
                .unwrap_or_else(|| Array1::zeros(current_state.len()));
            
            current_state = &current_state * (1.0 - regulation_strength) +
                          step * regulation_strength * 0.7 +
                          &hysteresis * regulation_strength * 0.3;
            
            // Update success tracking
            let step_success = if distance < 0.1 { 1.0 } else { 0.0 };
            success_count += step_success;
            total_steps += 1;
            
            // Store regulation attempt
            this.emotional_memory.push_back((
                Utc::now(),
                person_id.to_string(),
                current_state.clone(),
                regulation_strength
            ));
        }
        
        // Update strategy success rate
        let success_rate = if total_steps > 0 {
            success_count as f32 / total_steps as f32
        } else {
            0.0
        };
        
        Ok(current_state)
    }

    pub fn optimize_regulation_strategy(&mut self, strategy: &mut EmotionalRegulationStrategy, 
                                      transitions: &[EmotionalStateTransition]) -> Result<()> {
        // Analyze successful transitions
        let mut successful_transitions = Vec::new();
        for transition in transitions {
            if transition.success_rate > 0.7 {
                successful_transitions.push(transition);
            }
        }
        
        if !successful_transitions.is_empty() {
            // Find most efficient transition path
            let mut optimized_steps = Vec::new();
            let mut current_state = strategy.regulation_steps[0].clone();
            
            for transition in successful_transitions {
                if transition.transition_probability > 0.8 {
                    optimized_steps.push(transition.to_state.clone());
                }
            }
            
            // Update strategy with optimized steps
            strategy.regulation_steps = optimized_steps;
            
            // Recalculate metrics
            strategy.average_time = Duration::seconds((strategy.regulation_steps.len() * 2) as i64);
            strategy.energy_cost *= 0.8; // Assume 20% energy savings
        }
        
        Ok(())
    }

    pub fn train_pattern_model(&mut self, training_data: &[(Array1<f32>, Array1<f32>)]) -> Result<EmotionalPatternModel> {
        let input_dim = training_data[0].0.len();
        let output_dim = training_data[0].1.len();
        
        // Initialize model
        let mut model = EmotionalPatternModel {
            weights: Array2::from_shape_fn((output_dim, input_dim), |_| rand::random::<f32>() * 0.1),
            bias: Array1::zeros(output_dim),
            learning_rate: 0.01,
            epochs: 100,
            accuracy: 0.0,
            last_trained: Utc::now(),
        };
        
        // Training loop
        for epoch in 0..model.epochs {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;
            
            for (input, target) in training_data {
                // Forward pass
                let prediction = model.weights.dot(input) + &model.bias;
                
                // Calculate loss
                let error = target - &prediction;
                let loss = error.dot(&error);
                total_loss += loss;
                
                // Backward pass
                let gradient = -2.0 * error;
                model.weights = &model.weights - &(gradient.view().insert_axis(ndarray::Axis(1))
                    .dot(&input.view().insert_axis(ndarray::Axis(0))) * model.learning_rate);
                model.bias = &model.bias - &(gradient * model.learning_rate);
                
                // Track accuracy
                if loss < 0.1 {
                    correct_predictions += 1;
                }
            }
            
            // Update accuracy
            model.accuracy = correct_predictions as f32 / training_data.len() as f32;
            
            // Early stopping
            if model.accuracy > 0.95 {
                break;
            }
        }
        
        Ok(model)
    }

    pub fn predict_with_model(&self, model: &EmotionalPatternModel, input: &Array1<f32>) -> Result<Array1<f32>> {
        let prediction = model.weights.dot(input) + &model.bias;
        Ok(prediction)
    }

    pub fn visualize_emotional_state(&self, state: &Array1<f32>, context: &str) -> Result<EmotionalStateVisualization> {
        // Calculate dominant emotions
        let mut dominant_emotions = Vec::new();
        for (i, &value) in state.iter().enumerate() {
            if value.abs() > 0.3 {
                dominant_emotions.push((format!("emotion_{}", i), value));
            }
        }
        dominant_emotions.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        
        // Create intensity map
        let intensity_map: Vec<f32> = state.iter()
            .map(|&x| x.abs())
            .collect();
        
        // Get recent transitions
        let mut transition_path = Vec::new();
        let recent_memories: Vec<_> = this.emotional_memory
            .iter()
            .rev()
            .take(5)
            .collect();
        
        for (_, _, emotion, _) in recent_memories {
            transition_path.push(emotion.clone());
        }
        
        // Calculate stability
        let stability = if transition_path.len() > 1 {
            let mut total_change = 0.0;
            for window in transition_path.windows(2) {
                let change = (&window[1] - &window[0]).dot(&(&window[1] - &window[0])).sqrt();
                total_change += change;
            }
            1.0 - (total_change / (transition_path.len() - 1) as f32)
        } else {
            1.0
        };
        
        // Calculate context influence
        let mut context_influence = HashMap::new();
        if let Some(pattern) = this.emotional_patterns.get(&format!("{}:{}", "current", context)) {
            let influence = pattern.average_emotion.dot(state);
            context_influence.insert(context.to_string(), influence);
        }
        
        Ok(EmotionalStateVisualization {
            state_vector: state.clone(),
            dominant_emotions,
            intensity_map,
            transition_path,
            stability_score: stability,
            context_influence,
        })
    }

    pub fn generate_emotional_report(&self, visualization: &EmotionalStateVisualization) -> String {
        let mut report = String::new();
        
        report.push_str("Emotional State Report\n");
        report.push_str("=====================\n\n");
        
        report.push_str("Dominant Emotions:\n");
        for (emotion, intensity) in &visualization.dominant_emotions {
            report.push_str(&format!("- {}: {:.2}\n", emotion, intensity));
        }
        
        report.push_str("\nStability Score: ");
        report.push_str(&format!("{:.2}\n", visualization.stability_score));
        
        report.push_str("\nContext Influence:\n");
        for (context, influence) in &visualization.context_influence {
            report.push_str(&format!("- {}: {:.2}\n", context, influence));
        }
        
        report.push_str("\nRecent Emotional Trajectory:\n");
        for (i, state) in visualization.transition_path.iter().enumerate() {
            report.push_str(&format!("Step {}: {:?}\n", i, state));
        }
        
        report
    }

    pub fn update_model_with_feedback(&mut self, model: &mut EmotionalPatternModel, 
                                    input: &Array1<f32>, target: &Array1<f32>) -> Result<()> {
        // Calculate prediction error
        let prediction = model.weights.dot(input) + &model.bias;
        let error = target - &prediction;
        
        // Update weights and bias
        model.weights = &model.weights - &(error.view().insert_axis(ndarray::Axis(1))
            .dot(&input.view().insert_axis(ndarray::Axis(0))) * model.learning_rate);
        model.bias = &model.bias - &(error * model.learning_rate);
        
        // Update accuracy
        let loss = error.dot(&error);
        if loss < 0.1 {
            model.accuracy = (model.accuracy * 0.9 + 1.0 * 0.1).min(1.0);
        } else {
            model.accuracy = (model.accuracy * 0.9 + 0.0 * 0.1).max(0.0);
        }
        
        model.last_trained = Utc::now();
        
        Ok(())
    }

    pub fn analyze_emotional_components(&self, state: &Array1<f32>, history: &[(DateTime<Utc>, Array1<f32>)]) -> Result<Vec<EmotionalComponent>> {
        let mut components = Vec::new();
        
        // Analyze each emotional dimension
        for i in 0..state.len() {
            let mut component = EmotionalComponent {
                name: format!("emotion_{}", i),
                intensity: state[i].abs(),
                variance: 0.0,
                persistence: 0.0,
                influence: 0.0,
                related_components: Vec::new(),
            };
            
            // Calculate variance from history
            if !history.is_empty() {
                let values: Vec<f32> = history.iter().map(|(_, vec)| vec[i]).collect();
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                component.variance = values.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>() / values.len() as f32;
            }
            
            // Calculate persistence
            if history.len() > 1 {
                let mut same_sign_count = 0;
                for window in history.windows(2) {
                    if window[0].1[i].signum() == window[1].1[i].signum() {
                        same_sign_count += 1;
                    }
                }
                component.persistence = same_sign_count as f32 / (history.len() - 1) as f32;
            }
            
            // Calculate influence on other components
            let mut total_influence = 0.0;
            for j in 0..state.len() {
                if i != j {
                    let correlation = state[i] * state[j];
                    if correlation.abs() > 0.3 {
                        component.related_components.push(format!("emotion_{}", j));
                        total_influence += correlation.abs();
                    }
                }
            }
            component.influence = total_influence / (state.len() - 1) as f32;
            
            components.push(component);
        }
        
        Ok(components)
    }

    pub fn perform_detailed_analysis(&self, state: &Array1<f32>, context: &str) -> Result<EmotionalStateAnalysis> {
        // Get recent emotional history
        let history: Vec<_> = this.emotional_memory
            .iter()
            .rev()
            .take(10)
            .map(|(time, _, emotion, _)| (*time, emotion.clone()))
            .collect();
        
        // Analyze components
        let components = this.analyze_emotional_components(state, &history)?;
        
        // Identify primary and secondary emotions
        let mut primary_emotions = Vec::new();
        let mut secondary_emotions = Vec::new();
        
        for component in &components {
            if component.intensity > 0.5 {
                primary_emotions.push((component.name.clone(), component.intensity));
            } else if component.intensity > 0.2 {
                secondary_emotions.push((component.name.clone(), component.intensity));
            }
        }
        
        // Calculate overall metrics
        let emotional_intensity = state.dot(state).sqrt();
        let emotional_stability = 1.0 - components.iter()
            .map(|c| c.variance)
            .sum::<f32>() / components.len() as f32;
        
        let emotional_complexity = components.iter()
            .filter(|c| c.intensity > 0.2)
            .count() as f32 / components.len() as f32;
        
        let emotional_ambivalence = components.iter()
            .map(|c| (c.intensity * (1.0 - c.persistence)).abs())
            .sum::<f32>() / components.len() as f32;
        
        // Calculate emotional resonance
        let mut emotional_resonance = HashMap::new();
        for (person, pattern) in &this.emotional_patterns {
            let resonance = pattern.average_emotion.dot(state);
            if resonance.abs() > 0.3 {
                emotional_resonance.insert(person.clone(), resonance);
            }
        }
        
        // Calculate context influence
        let mut emotional_context = HashMap::new();
        if let Some(pattern) = this.emotional_patterns.get(&format!("{}:{}", "current", context)) {
            let influence = pattern.average_emotion.dot(state);
            emotional_context.insert(context.to_string(), influence);
        }
        
        // Calculate trends
        let mut emotional_trends = Vec::new();
        if history.len() > 1 {
            for i in 0..state.len() {
                let trend = (history[0].1[i] - history.last().unwrap().1[i]) / 
                          (history[0].0 - history.last().unwrap().0).num_seconds() as f32;
                if trend.abs() > 0.01 {
                    emotional_trends.push((format!("emotion_{}", i), trend));
                }
            }
        }
        
        Ok(EmotionalStateAnalysis {
            primary_emotions,
            secondary_emotions,
            emotional_intensity,
            emotional_stability,
            emotional_complexity,
            emotional_ambivalence,
            emotional_resonance,
            emotional_context,
            emotional_history: history,
            emotional_trends,
        })
    }

    pub fn optimize_regulation_strategy_with_analysis(&mut self, 
                                                    strategy: &mut EmotionalRegulationStrategy,
                                                    analysis: &EmotionalStateAnalysis) -> Result<()> {
        // Adjust strategy based on emotional complexity
        if analysis.emotional_complexity > 0.7 {
            // For complex emotional states, add more intermediate steps
            let additional_steps = (analysis.emotional_complexity * 5.0).ceil() as usize;
            strategy.regulation_steps = this.generate_additional_steps(
                &strategy.regulation_steps,
                additional_steps
            )?;
        }
        
        // Adjust energy cost based on stability
        strategy.energy_cost *= 1.0 + (1.0 - analysis.emotional_stability) * 0.5;
        
        // Adjust context specificity based on context influence
        let context_influence: f32 = analysis.emotional_context.values().sum();
        strategy.context_specificity = (strategy.context_specificity + context_influence).min(1.0);
        
        // Optimize steps based on emotional trends
        if !analysis.emotional_trends.is_empty() {
            strategy.regulation_steps = this.adjust_steps_for_trends(
                &strategy.regulation_steps,
                &analysis.emotional_trends
            )?;
        }
        
        // Update success rate based on analysis
        let stability_factor = analysis.emotional_stability * 0.4;
        let complexity_factor = (1.0 - analysis.emotional_complexity) * 0.3;
        let ambivalence_factor = (1.0 - analysis.emotional_ambivalence) * 0.3;
        
        strategy.success_rate = (stability_factor + complexity_factor + ambivalence_factor)
            .max(0.0).min(1.0);
        
        Ok(())
    }

    fn generate_additional_steps(&self, steps: &[Array1<f32>], additional: usize) -> Result<Vec<Array1<f32>>> {
        let mut new_steps = Vec::with_capacity(steps.len() + additional);
        new_steps.extend_from_slice(steps);
        
        for i in 0..additional {
            let idx = i % (steps.len() - 1);
            let step = &steps[idx] + &((&steps[idx + 1] - &steps[idx]) * 0.5);
            new_steps.insert(idx + 1, step);
        }
        
        Ok(new_steps)
    }

    fn adjust_steps_for_trends(&self, steps: &[Array1<f32>], trends: &[(String, f32)]) -> Result<Vec<Array1<f32>>> {
        let mut adjusted_steps = steps.to_vec();
        
        for (emotion, trend) in trends {
            if let Some(idx) = emotion.split('_').last().and_then(|s| s.parse::<usize>().ok()) {
                for step in &mut adjusted_steps {
                    // Adjust step based on trend direction
                    step[idx] += trend * 0.1;
                }
            }
        }
        
        Ok(adjusted_steps)
    }
}

#[derive(Debug, Clone)]
pub struct EmotionalTrend {
    pub trend_vector: Array1<f32>,
    pub volatility: f32,
    pub intensity: f32,
    pub sample_count: usize,
}

#[derive(Debug, Clone)]
pub struct EmotionalPrediction {
    pub predicted_emotion: Array1<f32>,
    pub confidence: f32,
    pub baseline_emotion: Array1<f32>,
    pub stimulus_impact: f32,
}

#[derive(Debug, Clone)]
pub struct EmotionalCorrelation {
    pub correlation_matrix: Array2<f32>,
    pub dominant_components: Vec<(usize, f32, Array1<f32>)>,
    pub pattern_strength: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vector_space_creation() {
        let space = VectorSpace::new(3).unwrap();
        assert_eq!(space.dimensions, 3);
        assert_eq!(space.basis.dim(), (3, 3));
    }

    #[test]
    fn test_interaction_matrix_stability() {
        let matrix = Array2::eye(3);
        let interaction = InteractionMatrix::new(matrix).unwrap();
        let stability = interaction.calculate_stability().unwrap();
        assert_relative_eq!(stability, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_markov_chain_properties() {
        let transition_matrix = Array2::eye(3);
        let chain = MarkovChain::new(transition_matrix).unwrap();
        let entropy = chain.calculate_entropy();
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_personality_system_integration() {
        let system = PersonalitySystem::new(3).unwrap();
        let stimulus = Array1::ones(3);
        let state = system.process_interaction("test", stimulus).unwrap();
        assert!(state.overall_stability >= 0.0 && state.overall_stability <= 1.0);
    }
} 