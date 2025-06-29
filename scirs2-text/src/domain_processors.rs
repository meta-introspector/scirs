//! Domain-specific text processors for specialized fields
//!
//! This module provides specialized text processing capabilities for different domains
//! including scientific, legal, and medical text processing with domain-specific
//! vocabularies, entity recognition, and preprocessing pipelines.

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::preprocess::{TextCleaner, TextNormalizer};
use crate::information_extraction::{Entity, EntityType, RuleBasedNER};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use lazy_static::lazy_static;

/// Domain-specific text processing domains
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Domain {
    /// Scientific and academic text
    Scientific,
    /// Legal documents and contracts
    Legal,
    /// Medical and clinical text
    Medical,
    /// Financial documents
    Financial,
    /// Patent documents
    Patent,
    /// News and journalism
    News,
    /// Social media content
    SocialMedia,
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Domain::Scientific => write!(f, "scientific"),
            Domain::Legal => write!(f, "legal"),
            Domain::Medical => write!(f, "medical"),
            Domain::Financial => write!(f, "financial"),
            Domain::Patent => write!(f, "patent"),
            Domain::News => write!(f, "news"),
            Domain::SocialMedia => write!(f, "social_media"),
        }
    }
}

/// Configuration for domain-specific processing
#[derive(Debug, Clone)]
pub struct DomainProcessorConfig {
    /// Target domain
    pub domain: Domain,
    /// Whether to preserve technical terms
    pub preserve_technical_terms: bool,
    /// Whether to normalize abbreviations
    pub normalize_abbreviations: bool,
    /// Whether to extract domain-specific entities
    pub extract_entities: bool,
    /// Whether to handle citations and references
    pub handle_citations: bool,
    /// Custom stop words for the domain
    pub custom_stop_words: HashSet<String>,
    /// Domain-specific regex patterns
    pub custom_patterns: HashMap<String, String>,
}

impl Default for DomainProcessorConfig {
    fn default() -> Self {
        Self {
            domain: Domain::Scientific,
            preserve_technical_terms: true,
            normalize_abbreviations: true,
            extract_entities: true,
            handle_citations: true,
            custom_stop_words: HashSet::new(),
            custom_patterns: HashMap::new(),
        }
    }
}

/// Scientific text processor
pub struct ScientificTextProcessor {
    config: DomainProcessorConfig,
    citation_regex: Regex,
    formula_regex: Regex,
    chemical_regex: Regex,
    measurement_regex: Regex,
    abbreviation_map: HashMap<String, String>,
    technical_terms: HashSet<String>,
}

impl ScientificTextProcessor {
    /// Create new scientific text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Scientific citation patterns
        let citation_regex = Regex::new(
            r"\(([A-Za-z]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?(?:;\s*[A-Za-z]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?)*)\)"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Mathematical formulas and equations
        let formula_regex = Regex::new(
            r"\$[^$]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]|\\begin\{[^}]+\}.*?\\end\{[^}]+\}"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Chemical formulas
        let chemical_regex = Regex::new(
            r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)*\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Scientific measurements
        let measurement_regex = Regex::new(
            r"\b\d+(?:\.\d+)?\s*(?:nm|μm|mm|cm|m|km|mg|g|kg|ml|l|°C|°F|K|Pa|kPa|MPa|Hz|kHz|MHz|GHz|V|mV|A|mA|Ω|W|kW|MW)\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Common scientific abbreviations
        let mut abbreviation_map = HashMap::new();
        abbreviation_map.insert("e.g.".to_string(), "for example".to_string());
        abbreviation_map.insert("i.e.".to_string(), "that is".to_string());
        abbreviation_map.insert("et al.".to_string(), "and others".to_string());
        abbreviation_map.insert("cf.".to_string(), "compare".to_string());
        abbreviation_map.insert("viz.".to_string(), "namely".to_string());
        
        // Technical terms to preserve
        let technical_terms = [
            "algorithm", "hypothesis", "methodology", "quantitative", "qualitative",
            "statistical", "correlation", "regression", "significance", "p-value",
            "standard deviation", "confidence interval", "meta-analysis", "systematic review",
        ].into_iter().map(|s| s.to_string()).collect();
        
        Ok(Self {
            config,
            citation_regex,
            formula_regex,
            chemical_regex,
            measurement_regex,
            abbreviation_map,
            technical_terms,
        })
    }
    
    /// Process scientific text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processed_text = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();
        
        // Extract citations
        if self.config.handle_citations {
            let citations = self.extract_citations(&processed_text)?;
            entities.extend(citations.into_iter().map(|citation| Entity {
                text: citation.clone(),
                entity_type: EntityType::Custom("citation".to_string()),
                confidence: 0.9,
                start_pos: 0, // Simplified
                end_pos: citation.len(),
            }));
        }
        
        // Extract and preserve formulas
        let formulas = self.extract_formulas(&processed_text)?;
        for (i, formula) in formulas.iter().enumerate() {
            let placeholder = format!("[FORMULA_{}]", i);
            processed_text = processed_text.replace(formula, &placeholder);
        }
        metadata.insert("formulas".to_string(), formulas.join("|"));
        
        // Extract measurements
        let measurements = self.extract_measurements(&processed_text)?;
        entities.extend(measurements.into_iter().map(|measurement| Entity {
            text: measurement.clone(),
            entity_type: EntityType::Custom("measurement".to_string()),
            confidence: 0.8,
            start_pos: 0,
            end_pos: measurement.len(),
        }));
        
        // Extract chemical formulas
        let chemicals = self.extract_chemicals(&processed_text)?;
        entities.extend(chemicals.into_iter().map(|chemical| Entity {
            text: chemical.clone(),
            entity_type: EntityType::Custom("chemical".to_string()),
            confidence: 0.7,
            start_pos: 0,
            end_pos: chemical.len(),
        }));
        
        // Normalize abbreviations
        if self.config.normalize_abbreviations {
            for (abbrev, expansion) in &self.abbreviation_map {
                processed_text = processed_text.replace(abbrev, expansion);
            }
        }
        
        // Clean text while preserving technical terms
        processed_text = self.clean_scientific_text(&processed_text)?;
        
        Ok(ProcessedDomainText {
            original_text: text.to_string(),
            processed_text,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }
    
    /// Extract citations from text
    fn extract_citations(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.citation_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Extract mathematical formulas
    fn extract_formulas(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.formula_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Extract chemical formulas
    fn extract_chemicals(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.chemical_regex
            .find_iter(text)
            .filter(|m| {
                let formula = m.as_str();
                // Basic heuristic to filter out non-chemical words
                formula.chars().any(|c| c.is_ascii_uppercase()) && 
                formula.chars().any(|c| c.is_ascii_digit())
            })
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Extract scientific measurements
    fn extract_measurements(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.measurement_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Clean scientific text while preserving important elements
    fn clean_scientific_text(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();
        
        // Remove excessive whitespace
        cleaned = Regex::new(r"\s+")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?
            .replace_all(&cleaned, " ")
            .to_string();
        
        // Normalize section headers
        cleaned = Regex::new(r"(?i)\b(abstract|introduction|methods?|results?|discussion|conclusion|references?)\s*:?\s*")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?
            .replace_all(&cleaned, |caps: &regex::Captures| {
                format!("[SECTION_{}] ", caps[1].to_uppercase())
            })
            .to_string();
        
        Ok(cleaned.trim().to_string())
    }
}

/// Legal text processor
pub struct LegalTextProcessor {
    config: DomainProcessorConfig,
    case_citation_regex: Regex,
    statute_regex: Regex,
    legal_terms: HashSet<String>,
    contract_clauses: Vec<String>,
}

impl LegalTextProcessor {
    /// Create new legal text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Legal case citations
        let case_citation_regex = Regex::new(
            r"\b[A-Z][a-zA-Z\s&,]+v\.?\s+[A-Z][a-zA-Z\s&,]+,?\s*\d+\s+[A-Z][a-z]*\.?\s*\d+(?:\s*\(\d+\))?"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Statute references
        let statute_regex = Regex::new(
            r"\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+|\b\d+\s+C\.?F\.?R\.?\s+§?\s*\d+"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Legal terminology
        let legal_terms = [
            "plaintiff", "defendant", "jurisdiction", "liability", "negligence",
            "contract", "tort", "damages", "injunction", "precedent", "statute",
            "constitutional", "federal", "state", "court", "judge", "jury",
            "evidence", "testimony", "witness", "counsel", "attorney", "lawyer",
        ].into_iter().map(|s| s.to_string()).collect();
        
        // Common contract clauses
        let contract_clauses = vec![
            "force majeure".to_string(),
            "indemnification".to_string(),
            "limitation of liability".to_string(),
            "intellectual property".to_string(),
            "confidentiality".to_string(),
            "termination".to_string(),
            "governing law".to_string(),
        ];
        
        Ok(Self {
            config,
            case_citation_regex,
            statute_regex,
            legal_terms,
            contract_clauses,
        })
    }
    
    /// Process legal text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processed_text = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();
        
        // Extract case citations
        let case_citations = self.extract_case_citations(&processed_text)?;
        entities.extend(case_citations.into_iter().map(|citation| Entity {
            text: citation.clone(),
            entity_type: EntityType::Custom("case_citation".to_string()),
            confidence: 0.9,
            start_pos: 0,
            end_pos: citation.len(),
        }));
        
        // Extract statute references
        let statutes = self.extract_statutes(&processed_text)?;
        entities.extend(statutes.into_iter().map(|statute| Entity {
            text: statute.clone(),
            entity_type: EntityType::Custom("statute".to_string()),
            confidence: 0.9,
            start_pos: 0,
            end_pos: statute.len(),
        }));
        
        // Identify contract clauses
        let clauses = self.identify_contract_clauses(&processed_text)?;
        metadata.insert("contract_clauses".to_string(), clauses.join("|"));
        
        // Normalize legal formatting
        processed_text = self.normalize_legal_text(&processed_text)?;
        
        Ok(ProcessedDomainText {
            original_text: text.to_string(),
            processed_text,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }
    
    /// Extract case citations
    fn extract_case_citations(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.case_citation_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Extract statute references
    fn extract_statutes(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.statute_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Identify contract clauses
    fn identify_contract_clauses(&self, text: &str) -> Result<Vec<String>> {
        let text_lower = text.to_lowercase();
        Ok(self.contract_clauses
            .iter()
            .filter(|clause| text_lower.contains(&clause.to_lowercase()))
            .cloned()
            .collect())
    }
    
    /// Normalize legal text formatting
    fn normalize_legal_text(&self, text: &str) -> Result<String> {
        let mut normalized = text.to_string();
        
        // Normalize section numbering
        normalized = Regex::new(r"\b(\d+)\.(\d+)\.(\d+)\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?
            .replace_all(&normalized, "Section $1.$2.$3")
            .to_string();
        
        // Normalize "whereas" clauses
        normalized = Regex::new(r"(?i)\bwhereas\b")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?
            .replace_all(&normalized, "WHEREAS")
            .to_string();
        
        Ok(normalized)
    }
}

/// Medical text processor
pub struct MedicalTextProcessor {
    config: DomainProcessorConfig,
    drug_regex: Regex,
    dosage_regex: Regex,
    symptom_regex: Regex,
    medical_terms: HashSet<String>,
    abbreviations: HashMap<String, String>,
}

impl MedicalTextProcessor {
    /// Create new medical text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Drug name patterns
        let drug_regex = Regex::new(
            r"\b[A-Z][a-z]+(?:mab|nib|tin|pine|pril|sartan|olol|azole|mycin|cillin)\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Dosage patterns
        let dosage_regex = Regex::new(
            r"\b\d+(?:\.\d+)?\s*(?:mg|g|ml|l|units?|tablets?|capsules?|cc)\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Symptom patterns (simplified)
        let symptom_regex = Regex::new(
            r"\b(?:pain|fever|nausea|headache|fatigue|cough|shortness of breath|chest pain)\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Medical terminology
        let medical_terms = [
            "diagnosis", "prognosis", "treatment", "therapy", "surgery", "procedure",
            "symptoms", "pathology", "etiology", "epidemiology", "pharmacology",
            "anatomy", "physiology", "clinical", "patient", "hospital", "physician",
        ].into_iter().map(|s| s.to_string()).collect();
        
        // Medical abbreviations
        let mut abbreviations = HashMap::new();
        abbreviations.insert("BP".to_string(), "blood pressure".to_string());
        abbreviations.insert("HR".to_string(), "heart rate".to_string());
        abbreviations.insert("RR".to_string(), "respiratory rate".to_string());
        abbreviations.insert("CBC".to_string(), "complete blood count".to_string());
        abbreviations.insert("ECG".to_string(), "electrocardiogram".to_string());
        abbreviations.insert("MRI".to_string(), "magnetic resonance imaging".to_string());
        abbreviations.insert("CT".to_string(), "computed tomography".to_string());
        
        Ok(Self {
            config,
            drug_regex,
            dosage_regex,
            symptom_regex,
            medical_terms,
            abbreviations,
        })
    }
    
    /// Process medical text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processed_text = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();
        
        // Extract drug names
        let drugs = self.extract_drugs(&processed_text)?;
        entities.extend(drugs.into_iter().map(|drug| Entity {
            text: drug.clone(),
            entity_type: EntityType::Custom("drug".to_string()),
            confidence: 0.8,
            start_pos: 0,
            end_pos: drug.len(),
        }));
        
        // Extract dosages
        let dosages = self.extract_dosages(&processed_text)?;
        entities.extend(dosages.into_iter().map(|dosage| Entity {
            text: dosage.clone(),
            entity_type: EntityType::Custom("dosage".to_string()),
            confidence: 0.9,
            start_pos: 0,
            end_pos: dosage.len(),
        }));
        
        // Extract symptoms
        let symptoms = self.extract_symptoms(&processed_text)?;
        metadata.insert("symptoms".to_string(), symptoms.join("|"));
        
        // Expand medical abbreviations
        if self.config.normalize_abbreviations {
            for (abbrev, expansion) in &self.abbreviations {
                let pattern = format(r"\b{}\b", regex::escape(abbrev));
                processed_text = Regex::new(&pattern)
                    .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?
                    .replace_all(&processed_text, expansion)
                    .to_string();
            }
        }
        
        // Clean medical text
        processed_text = self.clean_medical_text(&processed_text)?;
        
        Ok(ProcessedDomainText {
            original_text: text.to_string(),
            processed_text,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }
    
    /// Extract drug names
    fn extract_drugs(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.drug_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Extract dosage information
    fn extract_dosages(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.dosage_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Extract symptoms
    fn extract_symptoms(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.symptom_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Clean medical text
    fn clean_medical_text(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();
        
        // Normalize medical record formatting
        cleaned = Regex::new(r"(?i)\b(chief complaint|history of present illness|past medical history|medications|allergies|review of systems|physical examination|assessment|plan)\s*:?\s*")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?
            .replace_all(&cleaned, |caps: &regex::Captures| {
                format!("[{}] ", caps[1].to_uppercase().replace(" ", "_"))
            })
            .to_string();
        
        Ok(cleaned.trim().to_string())
    }
}

/// Financial text processor
pub struct FinancialTextProcessor {
    config: DomainProcessorConfig,
    currency_regex: Regex,
    financial_instrument_regex: Regex,
    percentage_regex: Regex,
    date_regex: Regex,
    financial_terms: HashSet<String>,
    currency_codes: HashSet<String>,
}

impl FinancialTextProcessor {
    /// Create new financial text processor
    pub fn new(config: DomainProcessorConfig) -> Result<Self> {
        // Currency patterns
        let currency_regex = Regex::new(
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?|\€\d+(?:,\d{3})*(?:\.\d{2})?|\£\d+(?:,\d{3})*(?:\.\d{2})?|USD\s*\d+|EUR\s*\d+|GBP\s*\d+"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Financial instruments
        let financial_instrument_regex = Regex::new(
            r"\b(?:bond|stock|share|equity|derivative|option|future|swap|ETF|mutual fund|hedge fund|REIT)\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Percentage patterns
        let percentage_regex = Regex::new(
            r"\b\d+(?:\.\d+)?%|percentage|percent|basis points?|bps\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Date patterns (financial context)
        let date_regex = Regex::new(
            r"\b(?:Q[1-4]|quarter)\s*\d{4}|\b\d{1,2}/\d{1,2}/\d{4}|\b\d{4}-\d{2}-\d{2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b"
        ).map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        
        // Financial terminology
        let financial_terms = [
            "revenue", "profit", "loss", "earnings", "dividend", "yield", "volatility",
            "liquidity", "capital", "assets", "liabilities", "equity", "debt", "credit",
            "investment", "portfolio", "risk", "return", "valuation", "margin", "leverage",
            "interest", "inflation", "gdp", "economic", "market", "trading", "broker",
        ].into_iter().map(|s| s.to_string()).collect();
        
        // Currency codes
        let currency_codes = [
            "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY", "INR", "BRL",
        ].into_iter().map(|s| s.to_string()).collect();
        
        Ok(Self {
            config,
            currency_regex,
            financial_instrument_regex,
            percentage_regex,
            date_regex,
            financial_terms,
            currency_codes,
        })
    }
    
    /// Process financial text
    pub fn process(&self, text: &str) -> Result<ProcessedDomainText> {
        let mut processed_text = text.to_string();
        let mut entities = Vec::new();
        let mut metadata = HashMap::new();
        
        // Extract currency amounts
        let currencies = self.extract_currencies(&processed_text)?;
        entities.extend(currencies.into_iter().map(|currency| Entity {
            text: currency.clone(),
            entity_type: EntityType::Custom("currency".to_string()),
            confidence: 0.9,
            start_pos: 0,
            end_pos: currency.len(),
        }));
        
        // Extract financial instruments
        let instruments = self.extract_financial_instruments(&processed_text)?;
        entities.extend(instruments.into_iter().map(|instrument| Entity {
            text: instrument.clone(),
            entity_type: EntityType::Custom("financial_instrument".to_string()),
            confidence: 0.8,
            start_pos: 0,
            end_pos: instrument.len(),
        }));
        
        // Extract percentages
        let percentages = self.extract_percentages(&processed_text)?;
        metadata.insert("percentages".to_string(), percentages.join("|"));
        
        // Extract financial dates
        let dates = self.extract_financial_dates(&processed_text)?;
        metadata.insert("financial_dates".to_string(), dates.join("|"));
        
        // Clean financial text
        processed_text = self.clean_financial_text(&processed_text)?;
        
        Ok(ProcessedDomainText {
            original_text: text.to_string(),
            processed_text,
            domain: self.config.domain.clone(),
            entities,
            metadata,
        })
    }
    
    /// Extract currency amounts
    fn extract_currencies(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.currency_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Extract financial instruments
    fn extract_financial_instruments(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.financial_instrument_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Extract percentages
    fn extract_percentages(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.percentage_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Extract financial dates
    fn extract_financial_dates(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.date_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }
    
    /// Clean financial text
    fn clean_financial_text(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();
        
        // Normalize financial section headers
        cleaned = Regex::new(r"(?i)\b(executive summary|financial highlights|income statement|balance sheet|cash flow|notes to financial statements)\s*:?\s*")
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?
            .replace_all(&cleaned, |caps: &regex::Captures| {
                format!("[{}] ", caps[1].to_uppercase().replace(" ", "_"))
            })
            .to_string();
        
        // Normalize currency symbols
        cleaned = cleaned.replace("$", "USD ");
        cleaned = cleaned.replace("€", "EUR ");
        cleaned = cleaned.replace("£", "GBP ");
        
        Ok(cleaned.trim().to_string())
    }
}

/// Result of domain-specific text processing
#[derive(Debug, Clone)]
pub struct ProcessedDomainText {
    /// Original input text
    pub original_text: String,
    /// Processed text
    pub processed_text: String,
    /// Domain type
    pub domain: Domain,
    /// Extracted domain-specific entities
    pub entities: Vec<Entity>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Unified domain processor that can handle multiple domains
pub struct UnifiedDomainProcessor {
    scientific_processor: Option<ScientificTextProcessor>,
    legal_processor: Option<LegalTextProcessor>,
    medical_processor: Option<MedicalTextProcessor>,
    financial_processor: Option<FinancialTextProcessor>,
}

impl UnifiedDomainProcessor {
    /// Create new unified domain processor
    pub fn new() -> Self {
        Self {
            scientific_processor: None,
            legal_processor: None,
            medical_processor: None,
            financial_processor: None,
        }
    }
    
    /// Add scientific text processing capability
    pub fn with_scientific(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.scientific_processor = Some(ScientificTextProcessor::new(config)?);
        Ok(self)
    }
    
    /// Add legal text processing capability
    pub fn with_legal(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.legal_processor = Some(LegalTextProcessor::new(config)?);
        Ok(self)
    }
    
    /// Add medical text processing capability
    pub fn with_medical(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.medical_processor = Some(MedicalTextProcessor::new(config)?);
        Ok(self)
    }
    
    /// Add financial text processing capability
    pub fn with_financial(mut self, config: DomainProcessorConfig) -> Result<Self> {
        self.financial_processor = Some(FinancialTextProcessor::new(config)?);
        Ok(self)
    }
    
    /// Process text for specified domain
    pub fn process(&self, text: &str, domain: Domain) -> Result<ProcessedDomainText> {
        match domain {
            Domain::Scientific => {
                if let Some(processor) = &self.scientific_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput("Scientific processor not configured".to_string()))
                }
            },
            Domain::Legal => {
                if let Some(processor) = &self.legal_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput("Legal processor not configured".to_string()))
                }
            },
            Domain::Medical => {
                if let Some(processor) = &self.medical_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput("Medical processor not configured".to_string()))
                }
            },
            Domain::Financial => {
                if let Some(processor) = &self.financial_processor {
                    processor.process(text)
                } else {
                    Err(TextError::InvalidInput("Financial processor not configured".to_string()))
                }
            },
            _ => Err(TextError::InvalidInput(format!("Domain {:?} not supported", domain))),
        }
    }
    
    /// Auto-detect domain from text content
    pub fn detect_domain(&self, text: &str) -> Domain {
        let text_lower = text.to_lowercase();
        
        // Simple heuristic-based domain detection
        let scientific_keywords = ["study", "research", "hypothesis", "methodology", "analysis", "results"];
        let legal_keywords = ["court", "law", "contract", "plaintiff", "defendant", "statute"];
        let medical_keywords = ["patient", "diagnosis", "treatment", "symptoms", "medication", "clinical"];
        let financial_keywords = ["revenue", "profit", "investment", "stock", "market", "financial", "earnings", "portfolio"];
        
        let sci_score = scientific_keywords.iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();
        
        let legal_score = legal_keywords.iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();
        
        let medical_score = medical_keywords.iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();
        
        let financial_score = financial_keywords.iter()
            .map(|&keyword| text_lower.matches(keyword).count())
            .sum::<usize>();
        
        let max_score = [sci_score, legal_score, medical_score, financial_score].iter().max().unwrap();
        
        if *max_score == 0 {
            Domain::Scientific // Default fallback
        } else if sci_score == *max_score {
            Domain::Scientific
        } else if legal_score == *max_score {
            Domain::Legal
        } else if medical_score == *max_score {
            Domain::Medical
        } else {
            Domain::Financial
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scientific_processor() {
        let config = DomainProcessorConfig {
            domain: Domain::Scientific,
            ..Default::default()
        };
        
        let processor = ScientificTextProcessor::new(config).unwrap();
        let text = "The results show (Smith et al., 2020) that H2O molecules at 25°C demonstrate significant behavior.";
        
        let result = processor.process(text).unwrap();
        assert_eq!(result.domain, Domain::Scientific);
        assert!(!result.entities.is_empty());
    }
    
    #[test]
    fn test_medical_processor() {
        let config = DomainProcessorConfig {
            domain: Domain::Medical,
            ..Default::default()
        };
        
        let processor = MedicalTextProcessor::new(config).unwrap();
        let text = "Patient reports chest pain and was prescribed 10mg aspirin.";
        
        let result = processor.process(text).unwrap();
        assert_eq!(result.domain, Domain::Medical);
        assert!(!result.entities.is_empty());
    }
    
    #[test]
    fn test_financial_processor() {
        let config = DomainProcessorConfig {
            domain: Domain::Financial,
            ..Default::default()
        };
        
        let processor = FinancialTextProcessor::new(config).unwrap();
        let text = "The company reported revenue of $1.2 million and stock price increased by 5.3%.";
        
        let result = processor.process(text).unwrap();
        assert_eq!(result.domain, Domain::Financial);
        assert!(!result.entities.is_empty());
        assert!(result.metadata.contains_key("percentages"));
    }
    
    #[test]
    fn test_domain_detection() {
        let processor = UnifiedDomainProcessor::new();
        
        let scientific_text = "This study analyzes the methodology used in the research hypothesis.";
        assert_eq!(processor.detect_domain(scientific_text), Domain::Scientific);
        
        let legal_text = "The court ruled that the defendant violated the contract law.";
        assert_eq!(processor.detect_domain(legal_text), Domain::Legal);
        
        let medical_text = "The patient was diagnosed with symptoms requiring clinical treatment.";
        assert_eq!(processor.detect_domain(medical_text), Domain::Medical);
        
        let financial_text = "The portfolio showed strong returns with profit margins increasing and market performance.";
        assert_eq!(processor.detect_domain(financial_text), Domain::Financial);
    }
}