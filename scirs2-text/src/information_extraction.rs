//! Information extraction utilities for structured data extraction from text
//!
//! This module provides tools for extracting structured information such as
//! named entities, key phrases, dates, and patterns from unstructured text.

use crate::error::Result;
use crate::tokenize::{Tokenizer, WordTokenizer};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use lazy_static::lazy_static;

lazy_static! {
    // Common regex patterns for information extraction
    static ref EMAIL_PATTERN: Regex = Regex::new(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ).unwrap();
    
    static ref URL_PATTERN: Regex = Regex::new(
        r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)"
    ).unwrap();
    
    static ref PHONE_PATTERN: Regex = Regex::new(
        r"(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})"
    ).unwrap();
    
    static ref DATE_PATTERN: Regex = Regex::new(
        r"\b(?:(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)?\d{2})|(?:(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01]))\b"
    ).unwrap();
    
    static ref TIME_PATTERN: Regex = Regex::new(
        r"\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?(?:\s*[aApP][mM])?\b"
    ).unwrap();
    
    static ref MONEY_PATTERN: Regex = Regex::new(
        r"[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:dollars?|euros?|pounds?|yen)"
    ).unwrap();
    
    static ref PERCENTAGE_PATTERN: Regex = Regex::new(
        r"\b\d+(?:\.\d+)?%\b"
    ).unwrap();
}

/// Entity types for named entity recognition
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    /// Person names and personal identifiers
    Person,
    /// Organization names, companies, institutions
    Organization,
    /// Geographic locations, places, addresses
    Location,
    /// Date expressions and temporal references
    Date,
    /// Time expressions and temporal references
    Time,
    /// Monetary amounts and currency references
    Money,
    /// Percentage values and ratios
    Percentage,
    /// Email addresses
    Email,
    /// URL and web addresses
    Url,
    /// Phone numbers and contact information
    Phone,
    /// Custom entity type defined by user
    Custom(String),
}

/// Extracted entity with type and position information
#[derive(Debug, Clone)]
pub struct Entity {
    /// The extracted text content
    pub text: String,
    /// The type of entity detected
    pub entity_type: EntityType,
    /// Start position in the original text
    pub start: usize,
    /// End position in the original text
    pub end: usize,
    /// Confidence score for the extraction (0.0 to 1.0)
    pub confidence: f64,
}

/// Simple rule-based named entity recognizer
pub struct RuleBasedNER {
    person_names: HashSet<String>,
    organizations: HashSet<String>,
    locations: HashSet<String>,
    custom_patterns: HashMap<String, Regex>,
}

impl RuleBasedNER {
    /// Create a new rule-based NER
    pub fn new() -> Self {
        Self {
            person_names: HashSet::new(),
            organizations: HashSet::new(),
            locations: HashSet::new(),
            custom_patterns: HashMap::new(),
        }
    }

    /// Add person names to the recognizer
    pub fn add_person_names<I: IntoIterator<Item = String>>(&mut self, names: I) {
        self.person_names.extend(names);
    }

    /// Add organization names
    pub fn add_organizations<I: IntoIterator<Item = String>>(&mut self, orgs: I) {
        self.organizations.extend(orgs);
    }

    /// Add location names
    pub fn add_locations<I: IntoIterator<Item = String>>(&mut self, locations: I) {
        self.locations.extend(locations);
    }

    /// Add custom pattern for entity extraction
    pub fn add_custom_pattern(&mut self, name: String, pattern: Regex) {
        self.custom_patterns.insert(name, pattern);
    }

    /// Extract entities from text
    pub fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Extract regex-based entities
        entities.extend(self.extract_pattern_entities(text, &EMAIL_PATTERN, EntityType::Email)?);
        entities.extend(self.extract_pattern_entities(text, &URL_PATTERN, EntityType::Url)?);
        entities.extend(self.extract_pattern_entities(text, &PHONE_PATTERN, EntityType::Phone)?);
        entities.extend(self.extract_pattern_entities(text, &DATE_PATTERN, EntityType::Date)?);
        entities.extend(self.extract_pattern_entities(text, &TIME_PATTERN, EntityType::Time)?);
        entities.extend(self.extract_pattern_entities(text, &MONEY_PATTERN, EntityType::Money)?);
        entities.extend(self.extract_pattern_entities(text, &PERCENTAGE_PATTERN, EntityType::Percentage)?);

        // Extract custom patterns
        for (name, pattern) in &self.custom_patterns {
            entities.extend(self.extract_pattern_entities(
                text,
                pattern,
                EntityType::Custom(name.clone())
            )?);
        }

        // Extract dictionary-based entities
        entities.extend(self.extract_dictionary_entities(text)?);

        // Sort by start position
        entities.sort_by_key(|e| e.start);

        Ok(entities)
    }

    /// Extract entities using regex patterns
    fn extract_pattern_entities(
        &self,
        text: &str,
        pattern: &Regex,
        entity_type: EntityType,
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        for mat in pattern.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(),
                entity_type: entity_type.clone(),
                start: mat.start(),
                end: mat.end(),
                confidence: 1.0, // High confidence for pattern matches
            });
        }

        Ok(entities)
    }

    /// Extract dictionary-based entities
    fn extract_dictionary_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let _text_lower = text.to_lowercase();

        // Simple word boundary matching for dictionary entities
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut position = 0;

        for word in words {
            let _word_lower = word.to_lowercase();
            let word_clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            
            if self.person_names.contains(word_clean) {
                if let Some(start) = text[position..].find(word) {
                    let abs_start = position + start;
                    entities.push(Entity {
                        text: word_clean.to_string(),
                        entity_type: EntityType::Person,
                        start: abs_start,
                        end: abs_start + word.len(),
                        confidence: 0.8,
                    });
                }
            } else if self.organizations.contains(word_clean) {
                if let Some(start) = text[position..].find(word) {
                    let abs_start = position + start;
                    entities.push(Entity {
                        text: word_clean.to_string(),
                        entity_type: EntityType::Organization,
                        start: abs_start,
                        end: abs_start + word.len(),
                        confidence: 0.8,
                    });
                }
            } else if self.locations.contains(word_clean) {
                if let Some(start) = text[position..].find(word) {
                    let abs_start = position + start;
                    entities.push(Entity {
                        text: word_clean.to_string(),
                        entity_type: EntityType::Location,
                        start: abs_start,
                        end: abs_start + word.len(),
                        confidence: 0.8,
                    });
                }
            }

            // Update position
            if let Some(next_pos) = text[position..].find(word) {
                position += next_pos + word.len();
            }
        }

        Ok(entities)
    }
}

impl Default for RuleBasedNER {
    fn default() -> Self {
        Self::new()
    }
}

/// Key phrase extractor using statistical methods
pub struct KeyPhraseExtractor {
    min_phrase_length: usize,
    max_phrase_length: usize,
    min_frequency: usize,
}

impl KeyPhraseExtractor {
    /// Create a new key phrase extractor
    pub fn new() -> Self {
        Self {
            min_phrase_length: 1,
            max_phrase_length: 3,
            min_frequency: 2,
        }
    }

    /// Set minimum phrase length
    pub fn with_min_length(mut self, length: usize) -> Self {
        self.min_phrase_length = length;
        self
    }

    /// Set maximum phrase length
    pub fn with_max_length(mut self, length: usize) -> Self {
        self.max_phrase_length = length;
        self
    }

    /// Set minimum frequency threshold
    pub fn with_min_frequency(mut self, freq: usize) -> Self {
        self.min_frequency = freq;
        self
    }

    /// Extract key phrases from text
    pub fn extract(&self, text: &str, tokenizer: &dyn Tokenizer) -> Result<Vec<(String, f64)>> {
        let tokens = tokenizer.tokenize(text)?;
        let mut phrase_counts: HashMap<String, usize> = HashMap::new();

        // Generate n-grams
        for n in self.min_phrase_length..=self.max_phrase_length {
            if tokens.len() >= n {
                for i in 0..=tokens.len() - n {
                    let phrase = tokens[i..i + n].join(" ");
                    *phrase_counts.entry(phrase).or_insert(0) += 1;
                }
            }
        }

        // Filter by frequency and calculate scores
        let mut phrases: Vec<(String, f64)> = phrase_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_frequency)
            .map(|(phrase, count)| {
                // Simple scoring: frequency * length
                let score = count as f64 * (phrase.split_whitespace().count() as f64).sqrt();
                (phrase, score)
            })
            .collect();

        // Sort by score descending
        phrases.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(phrases)
    }
}

impl Default for KeyPhraseExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern-based information extractor
pub struct PatternExtractor {
    patterns: Vec<(String, Regex)>,
}

impl PatternExtractor {
    /// Create a new pattern extractor
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a named pattern
    pub fn add_pattern(&mut self, name: String, pattern: Regex) {
        self.patterns.push((name, pattern));
    }

    /// Extract information matching patterns
    pub fn extract(&self, text: &str) -> Result<HashMap<String, Vec<String>>> {
        let mut results: HashMap<String, Vec<String>> = HashMap::new();

        for (name, pattern) in &self.patterns {
            let mut matches = Vec::new();
            
            for mat in pattern.find_iter(text) {
                matches.push(mat.as_str().to_string());
            }

            if !matches.is_empty() {
                results.insert(name.clone(), matches);
            }
        }

        Ok(results)
    }

    /// Extract with capture groups
    pub fn extract_with_groups(&self, text: &str) -> Result<HashMap<String, Vec<HashMap<String, String>>>> {
        let mut results: HashMap<String, Vec<HashMap<String, String>>> = HashMap::new();

        for (name, pattern) in &self.patterns {
            let mut matches = Vec::new();
            
            for caps in pattern.captures_iter(text) {
                let mut groups = HashMap::new();
                
                // Add full match
                if let Some(full_match) = caps.get(0) {
                    groups.insert("full".to_string(), full_match.as_str().to_string());
                }
                
                // Add numbered groups
                for i in 1..caps.len() {
                    if let Some(group) = caps.get(i) {
                        groups.insert(format!("group{}", i), group.as_str().to_string());
                    }
                }
                
                // Add named groups if any
                for name in pattern.capture_names().flatten() {
                    if let Some(group) = caps.name(name) {
                        groups.insert(name.to_string(), group.as_str().to_string());
                    }
                }
                
                matches.push(groups);
            }

            if !matches.is_empty() {
                results.insert(name.clone(), matches);
            }
        }

        Ok(results)
    }
}

impl Default for PatternExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Relation extractor for finding relationships between entities
pub struct RelationExtractor {
    relation_patterns: Vec<(String, Regex)>,
}

impl Default for RelationExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl RelationExtractor {
    /// Create a new relation extractor
    pub fn new() -> Self {
        Self {
            relation_patterns: Vec::new(),
        }
    }

    /// Add a relation pattern
    pub fn add_relation(&mut self, relation_type: String, pattern: Regex) {
        self.relation_patterns.push((relation_type, pattern));
    }

    /// Extract relations from text
    pub fn extract_relations(
        &self,
        text: &str,
        entities: &[Entity],
    ) -> Result<Vec<Relation>> {
        let mut relations = Vec::new();

        for (relation_type, pattern) in &self.relation_patterns {
            for caps in pattern.captures_iter(text) {
                if let Some(full_match) = caps.get(0) {
                    // Find entities that might be involved in this relation
                    let match_start = full_match.start();
                    let match_end = full_match.end();
                    
                    let involved_entities: Vec<&Entity> = entities
                        .iter()
                        .filter(|e| e.start >= match_start && e.end <= match_end)
                        .collect();

                    if involved_entities.len() >= 2 {
                        relations.push(Relation {
                            relation_type: relation_type.clone(),
                            subject: involved_entities[0].clone(),
                            object: involved_entities[1].clone(),
                            context: full_match.as_str().to_string(),
                            confidence: 0.7,
                        });
                    }
                }
            }
        }

        Ok(relations)
    }
}

/// Extracted relation between entities
#[derive(Debug, Clone)]
pub struct Relation {
    /// Type of relation (e.g., "works_for", "located_in")
    pub relation_type: String,
    /// Subject entity in the relation
    pub subject: Entity,
    /// Object entity in the relation
    pub object: Entity,
    /// Context text where the relation was found
    pub context: String,
    /// Confidence score for the relation extraction (0.0 to 1.0)
    pub confidence: f64,
}

/// Comprehensive information extraction pipeline
pub struct InformationExtractionPipeline {
    ner: RuleBasedNER,
    key_phrase_extractor: KeyPhraseExtractor,
    pattern_extractor: PatternExtractor,
    relation_extractor: RelationExtractor,
}

impl Default for InformationExtractionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl InformationExtractionPipeline {
    /// Create a new extraction pipeline
    pub fn new() -> Self {
        Self {
            ner: RuleBasedNER::new(),
            key_phrase_extractor: KeyPhraseExtractor::new(),
            pattern_extractor: PatternExtractor::new(),
            relation_extractor: RelationExtractor::new(),
        }
    }

    /// Set the NER component
    pub fn with_ner(mut self, ner: RuleBasedNER) -> Self {
        self.ner = ner;
        self
    }

    /// Set the key phrase extractor
    pub fn with_key_phrase_extractor(mut self, extractor: KeyPhraseExtractor) -> Self {
        self.key_phrase_extractor = extractor;
        self
    }

    /// Set the pattern extractor
    pub fn with_pattern_extractor(mut self, extractor: PatternExtractor) -> Self {
        self.pattern_extractor = extractor;
        self
    }

    /// Set the relation extractor
    pub fn with_relation_extractor(mut self, extractor: RelationExtractor) -> Self {
        self.relation_extractor = extractor;
        self
    }

    /// Extract all information from text
    pub fn extract(&self, text: &str) -> Result<ExtractedInformation> {
        let tokenizer = WordTokenizer::default();
        
        let entities = self.ner.extract_entities(text)?;
        let key_phrases = self.key_phrase_extractor.extract(text, &tokenizer)?;
        let patterns = self.pattern_extractor.extract(text)?;
        let relations = self.relation_extractor.extract_relations(text, &entities)?;

        Ok(ExtractedInformation {
            entities,
            key_phrases,
            patterns,
            relations,
        })
    }
}

/// Advanced temporal expression extractor
pub struct TemporalExtractor {
    patterns: Vec<(String, Regex)>,
}

impl Default for TemporalExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalExtractor {
    /// Create new temporal extractor with predefined patterns
    pub fn new() -> Self {
        let mut patterns = Vec::new();
        
        // Relative dates
        patterns.push((
            "relative_date".to_string(),
            Regex::new(r"(?i)\b(?:yesterday|today|tomorrow|last|next|this)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b").unwrap()
        ));
        
        // Time ranges
        patterns.push((
            "time_range".to_string(),
            Regex::new(r"(?i)\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9]\s*-\s*(?:[01]?[0-9]|2[0-3]):[0-5][0-9]\b").unwrap()
        ));
        
        // Durations
        patterns.push((
            "duration".to_string(),
            Regex::new(r"(?i)\b(?:\d+)\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b").unwrap()
        ));
        
        // Seasons and holidays
        patterns.push((
            "seasonal".to_string(),
            Regex::new(r"(?i)\b(?:spring|summer|fall|autumn|winter|christmas|thanksgiving|easter|halloween|new year)\b").unwrap()
        ));

        Self { patterns }
    }

    /// Extract temporal expressions from text
    pub fn extract(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        
        for (pattern_type, pattern) in &self.patterns {
            for mat in pattern.find_iter(text) {
                entities.push(Entity {
                    text: mat.as_str().to_string(),
                    entity_type: EntityType::Custom(format!("temporal_{}", pattern_type)),
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.85,
                });
            }
        }
        
        Ok(entities)
    }
}

/// Entity linker for connecting entities to knowledge bases
pub struct EntityLinker {
    knowledge_base: HashMap<String, KnowledgeBaseEntry>,
    alias_map: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct KnowledgeBaseEntry {
    pub canonical_name: String,
    pub entity_type: EntityType,
    pub aliases: Vec<String>,
    pub confidence: f64,
    pub metadata: HashMap<String, String>,
}

impl Default for EntityLinker {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityLinker {
    /// Create new entity linker
    pub fn new() -> Self {
        Self {
            knowledge_base: HashMap::new(),
            alias_map: HashMap::new(),
        }
    }

    /// Add entity to knowledge base
    pub fn add_entity(&mut self, entry: KnowledgeBaseEntry) {
        let canonical = entry.canonical_name.clone();
        
        // Add aliases to alias map
        for alias in &entry.aliases {
            self.alias_map.insert(alias.clone(), canonical.clone());
        }
        self.alias_map.insert(canonical.clone(), canonical.clone());
        
        self.knowledge_base.insert(canonical, entry);
    }

    /// Link extracted entities to knowledge base
    pub fn link_entities(&self, entities: &mut [Entity]) -> Result<Vec<LinkedEntity>> {
        let mut linked_entities = Vec::new();
        
        for entity in entities {
            if let Some(canonical_name) = self.alias_map.get(&entity.text.to_lowercase()) {
                if let Some(kb_entry) = self.knowledge_base.get(canonical_name) {
                    let confidence = entity.confidence * kb_entry.confidence;
                    linked_entities.push(LinkedEntity {
                        entity: entity.clone(),
                        canonical_name: kb_entry.canonical_name.clone(),
                        linked_confidence: confidence,
                        metadata: kb_entry.metadata.clone(),
                    });
                }
            }
        }
        
        Ok(linked_entities)
    }
}

/// Entity with knowledge base linking
#[derive(Debug, Clone)]
pub struct LinkedEntity {
    pub entity: Entity,
    pub canonical_name: String,
    pub linked_confidence: f64,
    pub metadata: HashMap<String, String>,
}

/// Coreference resolver for basic pronoun resolution
pub struct CoreferenceResolver {
    pronoun_patterns: Vec<Regex>,
}

impl Default for CoreferenceResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreferenceResolver {
    /// Create new coreference resolver
    pub fn new() -> Self {
        let pronoun_patterns = vec![
            Regex::new(r"\b(?i)(?:he|she|it|they|him|her|them|his|hers|its|their)\b").unwrap(),
            Regex::new(r"\b(?i)(?:this|that|these|those)\b").unwrap(),
            Regex::new(r"\b(?i)(?:the (?:company|organization|person|individual|entity))\b").unwrap(),
        ];
        
        Self { pronoun_patterns }
    }

    /// Resolve coreferences in text with entities
    pub fn resolve(&self, text: &str, entities: &[Entity]) -> Result<Vec<CoreferenceChain>> {
        let mut chains = Vec::new();
        let sentences = self.split_into_sentences(text);
        
        for (sent_idx, sentence) in sentences.iter().enumerate() {
            // Find entities in this sentence
            let sentence_entities: Vec<&Entity> = entities.iter()
                .filter(|e| text[e.start..e.end].trim() == sentence.trim() || 
                         sentence.contains(&e.text))
                .collect();
            
            // Find pronouns in this sentence
            for pattern in &self.pronoun_patterns {
                for mat in pattern.find_iter(sentence) {
                    // Try to resolve to nearest appropriate entity in previous sentences
                    if let Some(antecedent) = self.find_antecedent(
                        &mat.as_str().to_lowercase(),
                        &sentences[..sent_idx],
                        entities
                    ) {
                        chains.push(CoreferenceChain {
                            mentions: vec![
                                CoreferenceMention {
                                    text: antecedent.text.clone(),
                                    start: antecedent.start,
                                    end: antecedent.end,
                                    mention_type: MentionType::Entity,
                                },
                                CoreferenceMention {
                                    text: mat.as_str().to_string(),
                                    start: mat.start(),
                                    end: mat.end(),
                                    mention_type: MentionType::Pronoun,
                                }
                            ],
                            confidence: 0.6,
                        });
                    }
                }
            }
        }
        
        Ok(chains)
    }

    /// Split text into sentences (simple implementation)
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Find antecedent for a pronoun
    fn find_antecedent(&self, pronoun: &str, previous_sentences: &[String], entities: &[Entity]) -> Option<&Entity> {
        // Simple heuristic: find the closest person/organization entity
        let target_type = match pronoun {
            "he" | "him" | "his" => Some(EntityType::Person),
            "she" | "her" | "hers" => Some(EntityType::Person),
            "it" | "its" => Some(EntityType::Organization),
            "they" | "them" | "their" => None, // Could be either
            _ => None,
        };

        // Look for entities in reverse order (most recent first)
        for sentence in previous_sentences.iter().rev() {
            for entity in entities.iter().rev() {
                if sentence.contains(&entity.text) {
                    if let Some(expected_type) = &target_type {
                        if entity.entity_type == *expected_type {
                            return Some(entity);
                        }
                    } else {
                        // For ambiguous pronouns, return any person or organization
                        if matches!(entity.entity_type, EntityType::Person | EntityType::Organization) {
                            return Some(entity);
                        }
                    }
                }
            }
        }

        None
    }
}

/// Coreference chain representing linked mentions
#[derive(Debug, Clone)]
pub struct CoreferenceChain {
    pub mentions: Vec<CoreferenceMention>,
    pub confidence: f64,
}

/// Individual mention in a coreference chain
#[derive(Debug, Clone)]
pub struct CoreferenceMention {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub mention_type: MentionType,
}

/// Type of coreference mention
#[derive(Debug, Clone, PartialEq)]
pub enum MentionType {
    Entity,
    Pronoun,
    Description,
}

/// Advanced confidence scorer for entities
pub struct ConfidenceScorer {
    feature_weights: HashMap<String, f64>,
}

impl Default for ConfidenceScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceScorer {
    /// Create new confidence scorer
    pub fn new() -> Self {
        let mut feature_weights = HashMap::new();
        feature_weights.insert("pattern_match".to_string(), 0.3);
        feature_weights.insert("dictionary_match".to_string(), 0.2);
        feature_weights.insert("context_score".to_string(), 0.3);
        feature_weights.insert("length_score".to_string(), 0.1);
        feature_weights.insert("position_score".to_string(), 0.1);
        
        Self { feature_weights }
    }

    /// Calculate confidence score for an entity
    pub fn score_entity(&self, entity: &Entity, text: &str, context_window: usize) -> f64 {
        let mut features = HashMap::new();
        
        // Pattern match confidence (based on entity type)
        let pattern_score = match entity.entity_type {
            EntityType::Email | EntityType::Url | EntityType::Phone => 1.0,
            EntityType::Date | EntityType::Time | EntityType::Money | EntityType::Percentage => 0.9,
            _ => 0.7,
        };
        features.insert("pattern_match".to_string(), pattern_score);

        // Context score (surrounding words)
        let context_score = self.calculate_context_score(entity, text, context_window);
        features.insert("context_score".to_string(), context_score);

        // Length score (longer entities tend to be more reliable)
        let length_score = (entity.text.len() as f64 / 20.0).min(1.0);
        features.insert("length_score".to_string(), length_score);

        // Position score (entities at beginning/end might be more important)
        let position_score = if entity.start < text.len() / 4 || entity.end > 3 * text.len() / 4 {
            0.8
        } else {
            0.6
        };
        features.insert("position_score".to_string(), position_score);

        // Calculate weighted sum
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        
        for (feature, score) in features {
            if let Some(weight) = self.feature_weights.get(&feature) {
                total_score += score * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.5
        }
    }

    /// Calculate context score based on surrounding words
    fn calculate_context_score(&self, entity: &Entity, text: &str, window: usize) -> f64 {
        let start = entity.start.saturating_sub(window);
        let end = (entity.end + window).min(text.len());
        let context = &text[start..end];
        
        // Simple scoring based on presence of relevant keywords
        let keywords = match entity.entity_type {
            EntityType::Person => vec!["Mr.", "Ms.", "Dr.", "CEO", "President", "Manager"],
            EntityType::Organization => vec!["Inc.", "Corp.", "LLC", "Ltd.", "Company", "Foundation"],
            EntityType::Location => vec!["in", "at", "from", "to", "near", "City", "State"],
            EntityType::Money => vec!["cost", "price", "pay", "budget", "revenue", "profit"],
            EntityType::Date => vec!["on", "in", "during", "until", "since", "when"],
            _ => vec![],
        };

        let matches = keywords.iter()
            .filter(|&keyword| context.contains(keyword))
            .count();

        if keywords.is_empty() {
            0.5
        } else {
            (matches as f64 / keywords.len() as f64).min(1.0)
        }
    }
}

/// Enhanced information extraction pipeline with advanced features
pub struct AdvancedExtractionPipeline {
    ner: RuleBasedNER,
    key_phrase_extractor: KeyPhraseExtractor,
    pattern_extractor: PatternExtractor,
    relation_extractor: RelationExtractor,
    temporal_extractor: TemporalExtractor,
    entity_linker: EntityLinker,
    coreference_resolver: CoreferenceResolver,
    confidence_scorer: ConfidenceScorer,
}

impl Default for AdvancedExtractionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedExtractionPipeline {
    /// Create new advanced extraction pipeline
    pub fn new() -> Self {
        Self {
            ner: RuleBasedNER::new(),
            key_phrase_extractor: KeyPhraseExtractor::new(),
            pattern_extractor: PatternExtractor::new(),
            relation_extractor: RelationExtractor::new(),
            temporal_extractor: TemporalExtractor::new(),
            entity_linker: EntityLinker::new(),
            coreference_resolver: CoreferenceResolver::new(),
            confidence_scorer: ConfidenceScorer::new(),
        }
    }

    /// Configure components
    pub fn with_ner(mut self, ner: RuleBasedNER) -> Self {
        self.ner = ner;
        self
    }

    pub fn with_entity_linker(mut self, linker: EntityLinker) -> Self {
        self.entity_linker = linker;
        self
    }

    /// Extract comprehensive information with advanced features
    pub fn extract_advanced(&self, text: &str) -> Result<AdvancedExtractedInformation> {
        let tokenizer = WordTokenizer::default();
        
        // Basic extractions
        let mut entities = self.ner.extract_entities(text)?;
        let temporal_entities = self.temporal_extractor.extract(text)?;
        entities.extend(temporal_entities);
        
        // Enhance confidence scores
        for entity in &mut entities {
            entity.confidence = self.confidence_scorer.score_entity(entity, text, 50);
        }

        let key_phrases = self.key_phrase_extractor.extract(text, &tokenizer)?;
        let patterns = self.pattern_extractor.extract(text)?;
        let relations = self.relation_extractor.extract_relations(text, &entities)?;
        
        // Advanced extractions
        let linked_entities = self.entity_linker.link_entities(&mut entities)?;
        let coreference_chains = self.coreference_resolver.resolve(text, &entities)?;

        Ok(AdvancedExtractedInformation {
            entities,
            linked_entities,
            key_phrases,
            patterns,
            relations,
            coreference_chains,
        })
    }
}

/// Enhanced container for all extracted information
#[derive(Debug)]
pub struct AdvancedExtractedInformation {
    /// All entities extracted from the text
    pub entities: Vec<Entity>,
    /// Entities linked to knowledge base
    pub linked_entities: Vec<LinkedEntity>,
    /// Key phrases with importance scores
    pub key_phrases: Vec<(String, f64)>,
    /// Patterns found in the text organized by pattern type
    pub patterns: HashMap<String, Vec<String>>,
    /// Relations found between entities
    pub relations: Vec<Relation>,
    /// Coreference chains
    pub coreference_chains: Vec<CoreferenceChain>,
}

/// Container for all extracted information
#[derive(Debug)]
pub struct ExtractedInformation {
    /// All entities extracted from the text
    pub entities: Vec<Entity>,
    /// Key phrases with importance scores
    pub key_phrases: Vec<(String, f64)>,
    /// Patterns found in the text organized by pattern type
    pub patterns: HashMap<String, Vec<String>>,
    /// Relations found between entities
    pub relations: Vec<Relation>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_based_ner() {
        let mut ner = RuleBasedNER::new();
        ner.add_person_names(vec!["John".to_string(), "Jane".to_string()]);
        ner.add_organizations(vec!["Microsoft".to_string(), "Google".to_string()]);
        
        let text = "John works at Microsoft. His email is john@example.com";
        let entities = ner.extract_entities(text).unwrap();
        
        assert!(entities.len() >= 3); // John, Microsoft, email
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Person));
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Organization));
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Email));
    }

    #[test]
    fn test_key_phrase_extraction() {
        let extractor = KeyPhraseExtractor::new()
            .with_min_frequency(1)
            .with_max_length(2);
        
        let text = "machine learning is important. machine learning algorithms are complex.";
        let tokenizer = WordTokenizer::default();
        
        let phrases = extractor.extract(text, &tokenizer).unwrap();
        
        assert!(!phrases.is_empty());
        assert!(phrases.iter().any(|(p, _)| p.contains("machine learning")));
    }

    #[test]
    fn test_pattern_extraction() {
        let mut extractor = PatternExtractor::new();
        extractor.add_pattern(
            "price".to_string(),
            Regex::new(r"\$\d+(?:\.\d{2})?").unwrap()
        );
        
        let text = "The product costs $29.99 and shipping is $5.00";
        let results = extractor.extract(text).unwrap();
        
        assert!(results.contains_key("price"));
        assert_eq!(results["price"].len(), 2);
    }

    #[test]
    fn test_information_extraction_pipeline() {
        let pipeline = InformationExtractionPipeline::new();
        
        let text = "Apple Inc. announced that Tim Cook will visit London on January 15, 2024. Contact: info@apple.com";
        let info = pipeline.extract(text).unwrap();
        
        assert!(!info.entities.is_empty());
        assert!(info.entities.iter().any(|e| e.entity_type == EntityType::Email));
        assert!(info.entities.iter().any(|e| e.entity_type == EntityType::Date));
    }

    #[test]
    fn test_temporal_extractor() {
        let extractor = TemporalExtractor::new();
        
        let text = "The meeting is scheduled for next Monday from 2:00-4:00 PM. It will last 2 hours during winter season.";
        let entities = extractor.extract(text).unwrap();
        
        assert!(!entities.is_empty());
        assert!(entities.iter().any(|e| e.text.contains("next Monday")));
        assert!(entities.iter().any(|e| e.text.contains("2:00-4:00")));
        assert!(entities.iter().any(|e| e.text.contains("2 hours")));
        assert!(entities.iter().any(|e| e.text.contains("winter")));
    }

    #[test]
    fn test_entity_linker() {
        let mut linker = EntityLinker::new();
        
        // Add a knowledge base entry
        let kb_entry = KnowledgeBaseEntry {
            canonical_name: "Apple Inc.".to_string(),
            entity_type: EntityType::Organization,
            aliases: vec!["Apple".to_string(), "AAPL".to_string()],
            confidence: 0.9,
            metadata: HashMap::new(),
        };
        linker.add_entity(kb_entry);
        
        // Create test entities
        let mut entities = vec![
            Entity {
                text: "apple".to_string(),
                entity_type: EntityType::Organization,
                start: 0,
                end: 5,
                confidence: 0.7,
            }
        ];
        
        let linked = linker.link_entities(&mut entities).unwrap();
        assert!(!linked.is_empty());
        assert_eq!(linked[0].canonical_name, "Apple Inc.");
    }

    #[test]
    fn test_coreference_resolver() {
        let resolver = CoreferenceResolver::new();
        
        let entities = vec![
            Entity {
                text: "John Smith".to_string(),
                entity_type: EntityType::Person,
                start: 0,
                end: 10,
                confidence: 0.8,
            }
        ];
        
        let text = "John Smith is a CEO. He founded the company in 2020.";
        let chains = resolver.resolve(text, &entities).unwrap();
        
        // Should find a coreference chain for "He" -> "John Smith"
        assert!(!chains.is_empty());
        assert_eq!(chains[0].mentions.len(), 2);
    }

    #[test]
    fn test_confidence_scorer() {
        let scorer = ConfidenceScorer::new();
        
        let entity = Entity {
            text: "john@example.com".to_string(),
            entity_type: EntityType::Email,
            start: 20,
            end: 36,
            confidence: 0.5,
        };
        
        let text = "Please contact John at john@example.com for more information.";
        let score = scorer.score_entity(&entity, text, 10);
        
        // Email patterns should get high confidence
        assert!(score > 0.8);
    }

    #[test]
    fn test_advanced_extraction_pipeline() {
        let pipeline = AdvancedExtractionPipeline::new();
        
        let text = "Microsoft Corp. announced today that CEO Satya Nadella will visit New York next week. He will meet with partners.";
        let info = pipeline.extract_advanced(text).unwrap();
        
        // Should extract basic entities
        assert!(!info.entities.is_empty());
        
        // Should find temporal expressions
        assert!(info.entities.iter().any(|e| 
            matches!(e.entity_type, EntityType::Custom(ref s) if s.contains("temporal"))
        ));
        
        // Should have key phrases
        assert!(!info.key_phrases.is_empty());
    }

    #[test]
    fn test_context_scoring() {
        let scorer = ConfidenceScorer::new();
        
        // Test person entity with good context
        let person_entity = Entity {
            text: "Smith".to_string(),
            entity_type: EntityType::Person,
            start: 3,
            end: 8,
            confidence: 0.5,
        };
        
        let text_with_context = "Dr. Smith is the CEO of the company.";
        let score_with_context = scorer.score_entity(&person_entity, text_with_context, 10);
        
        let text_without_context = "The Smith family owns this.";
        let score_without_context = scorer.score_entity(&person_entity, text_without_context, 10);
        
        // Context with "Dr." and "CEO" should score higher
        assert!(score_with_context > score_without_context);
    }

    #[test]
    fn test_knowledge_base_aliases() {
        let mut linker = EntityLinker::new();
        
        let kb_entry = KnowledgeBaseEntry {
            canonical_name: "International Business Machines".to_string(),
            entity_type: EntityType::Organization,
            aliases: vec!["IBM".to_string(), "Big Blue".to_string()],
            confidence: 0.95,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("industry".to_string(), "Technology".to_string());
                meta
            },
        };
        linker.add_entity(kb_entry);
        
        let mut entities = vec![
            Entity {
                text: "ibm".to_string(), // lowercase
                entity_type: EntityType::Organization,
                start: 0,
                end: 3,
                confidence: 0.8,
            }
        ];
        
        let linked = linker.link_entities(&mut entities).unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].canonical_name, "International Business Machines");
        assert!(linked[0].metadata.contains_key("industry"));
    }

    #[test]
    fn test_sentence_splitting() {
        let resolver = CoreferenceResolver::new();
        let sentences = resolver.split_into_sentences("Hello world. How are you? Fine, thanks!");
        
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world");
        assert_eq!(sentences[1], "How are you");
        assert_eq!(sentences[2], "Fine, thanks");
    }
}