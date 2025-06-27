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
    Person,
    Organization,
    Location,
    Date,
    Time,
    Money,
    Percentage,
    Email,
    Url,
    Phone,
    Custom(String),
}

/// Extracted entity with type and position information
#[derive(Debug, Clone)]
pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
    pub start: usize,
    pub end: usize,
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
            
            if self.person_names.contains(&word_clean.to_string()) {
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
            } else if self.organizations.contains(&word_clean.to_string()) {
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
            } else if self.locations.contains(&word_clean.to_string()) {
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
    pub relation_type: String,
    pub subject: Entity,
    pub object: Entity,
    pub context: String,
    pub confidence: f64,
}

/// Comprehensive information extraction pipeline
pub struct InformationExtractionPipeline {
    ner: RuleBasedNER,
    key_phrase_extractor: KeyPhraseExtractor,
    pattern_extractor: PatternExtractor,
    relation_extractor: RelationExtractor,
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

/// Container for all extracted information
#[derive(Debug)]
pub struct ExtractedInformation {
    pub entities: Vec<Entity>,
    pub key_phrases: Vec<(String, f64)>,
    pub patterns: HashMap<String, Vec<String>>,
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
}