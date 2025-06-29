//! Hugging Face compatibility layer for interoperability
//!
//! This module provides compatibility interfaces and adapters to work with
//! Hugging Face model formats, tokenizers, and APIs, enabling seamless
//! integration with the broader ML ecosystem.

use crate::error::{Result, TextError};
use crate::model_registry::{ModelMetadata, ModelRegistry, ModelType, RegistrableModel, SerializableModelData};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::transformer::{TransformerConfig, TransformerModel};
use crate::vectorize::{CountVectorizer, TfidfVectorizer};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use std::io::{BufReader, BufWriter};

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};

/// Hugging Face model configuration format
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct HfConfig {
    /// Model architecture type
    pub architectures: Vec<String>,
    /// Model type (e.g., "bert", "gpt2", "roberta")
    pub model_type: String,
    /// Number of attention heads
    pub num_attention_heads: Option<usize>,
    /// Hidden size
    pub hidden_size: Option<usize>,
    /// Intermediate size
    pub intermediate_size: Option<usize>,
    /// Number of hidden layers
    pub num_hidden_layers: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Maximum position embeddings
    pub max_position_embeddings: Option<usize>,
    /// Additional configuration parameters
    pub extra_config: HashMap<String, serde_json::Value>,
}

impl Default for HfConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["BertModel".to_string()],
            model_type: "bert".to_string(),
            num_attention_heads: Some(12),
            hidden_size: Some(768),
            intermediate_size: Some(3072),
            num_hidden_layers: Some(12),
            vocab_size: Some(30522),
            max_position_embeddings: Some(512),
            extra_config: HashMap::new(),
        }
    }
}

impl HfConfig {
    /// Convert to SciRS2 transformer config
    pub fn to_transformer_config(&self) -> Result<TransformerConfig> {
        Ok(TransformerConfig {
            d_model: self.hidden_size.unwrap_or(768),
            n_heads: self.num_attention_heads.unwrap_or(12),
            d_ff: self.intermediate_size.unwrap_or(3072),
            n_encoder_layers: self.num_hidden_layers.unwrap_or(12),
            n_decoder_layers: self.num_hidden_layers.unwrap_or(12),
            max_seq_len: self.max_position_embeddings.unwrap_or(512),
            dropout: 0.1,
            vocab_size: self.vocab_size.unwrap_or(30522),
        })
    }
    
    /// Create from transformer config
    pub fn from_transformer_config(config: &TransformerConfig) -> Self {
        Self {
            architectures: vec!["TransformerModel".to_string()],
            model_type: "transformer".to_string(),
            num_attention_heads: Some(config.n_heads),
            hidden_size: Some(config.d_model),
            intermediate_size: Some(config.d_ff),
            num_hidden_layers: Some(config.n_encoder_layers),
            vocab_size: Some(config.vocab_size),
            max_position_embeddings: Some(config.max_seq_len),
            extra_config: HashMap::new(),
        }
    }
}

/// Hugging Face tokenizer configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct HfTokenizerConfig {
    /// Tokenizer type
    pub tokenizer_type: String,
    /// Vocabulary file path
    pub vocab_file: Option<PathBuf>,
    /// Merges file path (for BPE)
    pub merges_file: Option<PathBuf>,
    /// Special tokens
    pub special_tokens: HashMap<String, String>,
    /// Maximum sequence length
    pub max_len: usize,
    /// Padding token
    pub pad_token: String,
    /// Unknown token
    pub unk_token: String,
    /// Start of sequence token
    pub bos_token: Option<String>,
    /// End of sequence token
    pub eos_token: Option<String>,
}

impl Default for HfTokenizerConfig {
    fn default() -> Self {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("[CLS]".to_string(), "cls_token".to_string());
        special_tokens.insert("[SEP]".to_string(), "sep_token".to_string());
        special_tokens.insert("[PAD]".to_string(), "pad_token".to_string());
        special_tokens.insert("[UNK]".to_string(), "unk_token".to_string());
        special_tokens.insert("[MASK]".to_string(), "mask_token".to_string());
        
        Self {
            tokenizer_type: "WordPiece".to_string(),
            vocab_file: None,
            merges_file: None,
            special_tokens,
            max_len: 512,
            pad_token: "[PAD]".to_string(),
            unk_token: "[UNK]".to_string(),
            bos_token: Some("[CLS]".to_string()),
            eos_token: Some("[SEP]".to_string()),
        }
    }
}

/// Hugging Face compatible tokenizer wrapper
pub struct HfTokenizer {
    /// Underlying tokenizer
    tokenizer: Box<dyn Tokenizer>,
    /// Tokenizer configuration
    config: HfTokenizerConfig,
    /// Vocabulary mapping
    vocab: HashMap<String, usize>,
    /// Reverse vocabulary mapping
    reverse_vocab: HashMap<usize, String>,
}

impl HfTokenizer {
    /// Create new HF-compatible tokenizer
    pub fn new(tokenizer: Box<dyn Tokenizer>, config: HfTokenizerConfig) -> Self {
        // Create basic vocabulary (in practice, this would be loaded from files)
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        // Add special tokens
        let mut token_id = 0;
        for (token, _) in &config.special_tokens {
            vocab.insert(token.clone(), token_id);
            reverse_vocab.insert(token_id, token.clone());
            token_id += 1;
        }
        
        Self {
            tokenizer,
            config,
            vocab,
            reverse_vocab,
        }
    }
    
    /// Tokenize text with HF-compatible output
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<HfEncodedInput> {
        let mut tokens = self.tokenizer.tokenize(text);
        
        // Add special tokens if requested
        if add_special_tokens {
            if let Some(bos_token) = &self.config.bos_token {
                tokens.insert(0, bos_token.clone());
            }
            if let Some(eos_token) = &self.config.eos_token {
                tokens.push(eos_token.clone());
            }
        }
        
        // Convert tokens to IDs
        let input_ids: Vec<usize> = tokens.iter()
            .map(|token| self.vocab.get(token).copied().unwrap_or(
                self.vocab.get(&self.config.unk_token).copied().unwrap_or(0)
            ))
            .collect();
        
        // Create attention mask (1 for real tokens, 0 for padding)
        let attention_mask = vec![1; input_ids.len()];
        
        // Token type IDs (all 0 for single sentence)
        let token_type_ids = vec![0; input_ids.len()];
        
        Ok(HfEncodedInput {
            input_ids,
            attention_mask,
            token_type_ids: Some(token_type_ids),
            tokens,
        })
    }
    
    /// Batch encode multiple texts
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<HfEncodedInput>> {
        texts.iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> Result<String> {
        let tokens: Vec<String> = token_ids.iter()
            .filter_map(|&id| self.reverse_vocab.get(&id))
            .filter(|token| {
                if skip_special_tokens {
                    !self.config.special_tokens.contains_key(*token)
                } else {
                    true
                }
            })
            .cloned()
            .collect();
        
        Ok(tokens.join(" "))
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// HF-compatible encoded input format
#[derive(Debug, Clone)]
pub struct HfEncodedInput {
    /// Token IDs
    pub input_ids: Vec<usize>,
    /// Attention mask
    pub attention_mask: Vec<i32>,
    /// Token type IDs (for multi-sentence tasks)
    pub token_type_ids: Option<Vec<usize>>,
    /// Original tokens
    pub tokens: Vec<String>,
}

/// Hugging Face model adapter
pub struct HfModelAdapter {
    /// Model configuration
    config: HfConfig,
    /// Model registry for storage
    registry: Option<ModelRegistry>,
    /// Model metadata
    metadata: Option<ModelMetadata>,
}

impl HfModelAdapter {
    /// Create new HF model adapter
    pub fn new(config: HfConfig) -> Self {
        Self {
            config,
            registry: None,
            metadata: None,
        }
    }
    
    /// Set model registry
    pub fn with_registry(mut self, registry: ModelRegistry) -> Self {
        self.registry = Some(registry);
        self
    }
    
    /// Set model metadata
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
    
    /// Load model from HF format directory
    pub fn load_from_hf_directory<P: AsRef<Path>>(&self, model_path: P) -> Result<TransformerModel> {
        let model_path = model_path.as_ref();
        
        // Check for required files
        let config_file = model_path.join("config.json");
        if !config_file.exists() {
            return Err(TextError::InvalidInput(
                "HF config.json not found".to_string()
            ));
        }
        
        // Load configuration
        let transformer_config = if config_file.exists() {
            #[cfg(feature = "serde-support")]
            {
                let file = fs::File::open(&config_file)
                    .map_err(|e| TextError::IoError(format!("Failed to open config file: {}", e)))?;
                let reader = BufReader::new(file);
                let hf_config: HfConfig = serde_json::from_reader(reader)
                    .map_err(|e| TextError::InvalidInput(format!("Failed to deserialize config: {}", e)))?;
                hf_config.to_transformer_config()?
            }
            
            #[cfg(not(feature = "serde-support"))]
            {
                // Fallback when serde is not available
                self.config.to_transformer_config()?
            }
        } else {
            self.config.to_transformer_config()?
        };
        
        // Create vocabulary (simplified - would load from tokenizer.json)
        let vocabulary: Vec<String> = (0..transformer_config.vocab_size)
            .map(|i| format!("[TOKEN_{}]", i))
            .collect();
        
        // Create transformer model
        TransformerModel::new(transformer_config, vocabulary)
    }
    
    /// Save model to HF format directory
    pub fn save_to_hf_directory<P: AsRef<Path>>(
        &self,
        model: &TransformerModel,
        output_path: P,
    ) -> Result<()> {
        let output_path = output_path.as_ref();
        
        // Create output directory
        std::fs::create_dir_all(output_path)
            .map_err(|e| TextError::IoError(format!("Failed to create directory: {}", e)))?;
        
        // Save configuration
        #[cfg(feature = "serde-support")]
        {
            let config_file = fs::File::create(output_path.join("config.json"))
                .map_err(|e| TextError::IoError(format!("Failed to create config file: {}", e)))?;
            let writer = BufWriter::new(config_file);
            serde_json::to_writer_pretty(writer, &self.config)
                .map_err(|e| TextError::InvalidInput(format!("Failed to serialize config: {}", e)))?;
        }
        
        #[cfg(not(feature = "serde-support"))]
        {
            let config_json = format!("{:#?}", self.config);
            fs::write(output_path.join("config.json"), config_json)
                .map_err(|e| TextError::IoError(format!("Failed to write config: {}", e)))?;
        }
        
        // Save model weights (simplified - in practice would save actual model weights)
        let model_data = b"placeholder_model_weights";
        fs::write(output_path.join("pytorch_model.bin"), model_data)
            .map_err(|e| TextError::IoError(format!("Failed to write model: {}", e)))?;
        
        // Save tokenizer configuration
        let tokenizer_config = HfTokenizerConfig::default();
        
        #[cfg(feature = "serde-support")]
        {
            let tokenizer_file = fs::File::create(output_path.join("tokenizer.json"))
                .map_err(|e| TextError::IoError(format!("Failed to create tokenizer file: {}", e)))?;
            let writer = BufWriter::new(tokenizer_file);
            serde_json::to_writer_pretty(writer, &tokenizer_config)
                .map_err(|e| TextError::InvalidInput(format!("Failed to serialize tokenizer config: {}", e)))?;
        }
        
        #[cfg(not(feature = "serde-support"))]
        {
            let tokenizer_json = format!("{:#?}", tokenizer_config);
            fs::write(output_path.join("tokenizer.json"), tokenizer_json)
                .map_err(|e| TextError::IoError(format!("Failed to write tokenizer: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Create HF-compatible pipeline
    pub fn create_pipeline(&self, task: &str) -> Result<HfPipeline> {
        match task {
            "text-classification" => Ok(HfPipeline::TextClassification(
                TextClassificationPipeline::new()
            )),
            "feature-extraction" => Ok(HfPipeline::FeatureExtraction(
                FeatureExtractionPipeline::new()
            )),
            "fill-mask" => Ok(HfPipeline::FillMask(
                FillMaskPipeline::new()
            )),
            "zero-shot-classification" => Ok(HfPipeline::ZeroShotClassification(
                ZeroShotClassificationPipeline::new()
            )),
            "question-answering" => Ok(HfPipeline::QuestionAnswering(
                QuestionAnsweringPipeline::new()
            )),
            _ => Err(TextError::InvalidInput(format!("Unsupported task: {}", task))),
        }
    }
}

/// HF-compatible pipeline types
#[derive(Debug)]
pub enum HfPipeline {
    /// Text classification pipeline
    TextClassification(TextClassificationPipeline),
    /// Feature extraction pipeline
    FeatureExtraction(FeatureExtractionPipeline),
    /// Fill mask pipeline
    FillMask(FillMaskPipeline),
    /// Zero-shot classification pipeline
    ZeroShotClassification(ZeroShotClassificationPipeline),
    /// Question answering pipeline
    QuestionAnswering(QuestionAnsweringPipeline),
}

/// Text classification pipeline
#[derive(Debug)]
pub struct TextClassificationPipeline {
    /// Labels for classification
    labels: Vec<String>,
}

impl TextClassificationPipeline {
    /// Create new text classification pipeline
    pub fn new() -> Self {
        Self {
            labels: vec!["NEGATIVE".to_string(), "POSITIVE".to_string()],
        }
    }
    
    /// Run classification on text
    pub fn predict(&self, text: &str) -> Result<Vec<ClassificationResult>> {
        // Simplified prediction (would use actual model)
        let score = text.len() as f64 / 100.0; // Dummy score
        let predicted_label = if score > 0.5 { "POSITIVE" } else { "NEGATIVE" };
        
        Ok(vec![ClassificationResult {
            label: predicted_label.to_string(),
            score: score.min(1.0),
        }])
    }
}

/// Feature extraction pipeline
#[derive(Debug)]
pub struct FeatureExtractionPipeline;

impl FeatureExtractionPipeline {
    /// Create new feature extraction pipeline
    pub fn new() -> Self {
        Self
    }
    
    /// Extract features from text
    pub fn extract_features(&self, text: &str) -> Result<Array2<f64>> {
        // Dummy feature extraction (would use actual model)
        let feature_dim = 768;
        let seq_len = text.split_whitespace().count().max(1);
        
        Ok(Array2::from_shape_fn((seq_len, feature_dim), |_| {
            rand::random::<f64>()
        }))
    }
}

/// Fill mask pipeline
#[derive(Debug)]
pub struct FillMaskPipeline;

impl FillMaskPipeline {
    /// Create new fill mask pipeline
    pub fn new() -> Self {
        Self
    }
    
    /// Fill masked tokens in text
    pub fn fill_mask(&self, text: &str) -> Result<Vec<FillMaskResult>> {
        // Simplified mask filling (would use actual model)
        if text.contains("[MASK]") {
            Ok(vec![
                FillMaskResult {
                    sequence: text.replace("[MASK]", "the"),
                    score: 0.8,
                    token: "the".to_string(),
                    token_id: 1,
                },
                FillMaskResult {
                    sequence: text.replace("[MASK]", "a"),
                    score: 0.15,
                    token: "a".to_string(),
                    token_id: 2,
                },
            ])
        } else {
            Err(TextError::InvalidInput("No [MASK] token found".to_string()))
        }
    }
}

/// Classification result
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Predicted label
    pub label: String,
    /// Confidence score
    pub score: f64,
}

/// Fill mask result
#[derive(Debug, Clone)]
pub struct FillMaskResult {
    /// Completed sequence
    pub sequence: String,
    /// Confidence score
    pub score: f64,
    /// Predicted token
    pub token: String,
    /// Token ID
    pub token_id: usize,
}

/// Zero-shot classification pipeline
#[derive(Debug)]
pub struct ZeroShotClassificationPipeline {
    /// Hypothesis template
    hypothesis_template: String,
}

impl ZeroShotClassificationPipeline {
    /// Create new zero-shot classification pipeline
    pub fn new() -> Self {
        Self {
            hypothesis_template: "This example is {}.".to_string(),
        }
    }
    
    /// Classify text against multiple labels
    pub fn classify(&self, text: &str, candidate_labels: &[&str]) -> Result<Vec<ClassificationResult>> {
        let mut results = Vec::new();
        
        // Simplified zero-shot classification (would use actual NLI model)
        for (i, &label) in candidate_labels.iter().enumerate() {
            let score = (text.len() as f64 + label.len() as f64) / 200.0; // Dummy score
            let score = score.min(1.0).max(0.0);
            
            results.push(ClassificationResult {
                label: label.to_string(),
                score,
            });
        }
        
        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(results)
    }
    
    /// Set hypothesis template
    pub fn set_hypothesis_template(&mut self, template: String) {
        self.hypothesis_template = template;
    }
}

/// Question answering pipeline
#[derive(Debug)]
pub struct QuestionAnsweringPipeline;

impl QuestionAnsweringPipeline {
    /// Create new question answering pipeline
    pub fn new() -> Self {
        Self
    }
    
    /// Answer question based on context
    pub fn answer(&self, question: &str, context: &str) -> Result<QuestionAnsweringResult> {
        // Simplified QA (would use actual QA model)
        let context_words: Vec<&str> = context.split_whitespace().collect();
        let question_lower = question.to_lowercase();
        
        // Find potential answer span (very simplified)
        let start = if question_lower.contains("what") || question_lower.contains("who") {
            context_words.len() / 4
        } else if question_lower.contains("when") {
            context_words.len() / 3
        } else {
            context_words.len() / 2
        };
        
        let end = (start + 3).min(context_words.len());
        let answer = context_words[start..end].join(" ");
        
        Ok(QuestionAnsweringResult {
            answer,
            score: 0.8,
            start,
            end,
        })
    }
}

/// Question answering result
#[derive(Debug, Clone)]
pub struct QuestionAnsweringResult {
    /// The answer text
    pub answer: String,
    /// Confidence score
    pub score: f64,
    /// Start position in context
    pub start: usize,
    /// End position in context
    pub end: usize,
}

/// Hugging Face Hub integration utilities
pub struct HfHub;

impl HfHub {
    /// Download model from Hugging Face Hub (placeholder)
    pub fn download_model(model_id: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
        let cache_dir = cache_dir.unwrap_or(Path::new("~/.cache/huggingface"));
        let model_path = cache_dir.join(model_id);
        
        // In a real implementation, this would download from HF Hub
        if !model_path.exists() {
            std::fs::create_dir_all(&model_path)
                .map_err(|e| TextError::IoError(format!("Failed to create cache dir: {}", e)))?;
        }
        
        Ok(model_path)
    }
    
    /// List available models (placeholder)
    pub fn list_models(filter_task: Option<&str>) -> Result<Vec<String>> {
        // Placeholder model list
        let models = vec![
            "bert-base-uncased".to_string(),
            "roberta-base".to_string(),
            "gpt2".to_string(),
            "distilbert-base-uncased".to_string(),
        ];
        
        Ok(models)
    }
    
    /// Get model info (placeholder)
    pub fn model_info(model_id: &str) -> Result<HashMap<String, String>> {
        let mut info = HashMap::new();
        info.insert("model_id".to_string(), model_id.to_string());
        info.insert("pipeline_tag".to_string(), "text-classification".to_string());
        info.insert("library_name".to_string(), "transformers".to_string());
        
        Ok(info)
    }
}

/// Convert between HF and SciRS2 formats
pub struct FormatConverter;

impl FormatConverter {
    /// Convert HF config to SciRS2 transformer config
    pub fn hf_to_scirs2_config(hf_config: &HfConfig) -> Result<TransformerConfig> {
        hf_config.to_transformer_config()
    }
    
    /// Convert SciRS2 transformer config to HF config
    pub fn scirs2_to_hf_config(scirs2_config: &TransformerConfig) -> HfConfig {
        HfConfig::from_transformer_config(scirs2_config)
    }
    
    /// Convert HF tokenizer output to SciRS2 format
    pub fn hf_to_scirs2_tokens(hf_encoded: &HfEncodedInput) -> Vec<String> {
        hf_encoded.tokens.clone()
    }
    
    /// Convert SciRS2 tokens to HF format
    pub fn scirs2_to_hf_tokens(tokens: &[String]) -> HfEncodedInput {
        let input_ids: Vec<usize> = (0..tokens.len()).collect();
        let attention_mask = vec![1; tokens.len()];
        
        HfEncodedInput {
            input_ids,
            attention_mask,
            token_type_ids: Some(vec![0; tokens.len()]),
            tokens: tokens.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::WordTokenizer;
    
    #[test]
    fn test_hf_config_conversion() {
        let hf_config = HfConfig::default();
        let transformer_config = hf_config.to_transformer_config().unwrap();
        
        assert_eq!(transformer_config.d_model, 768);
        assert_eq!(transformer_config.n_heads, 12);
        assert_eq!(transformer_config.vocab_size, 30522);
    }
    
    #[test]
    fn test_hf_tokenizer() {
        let word_tokenizer = Box::new(WordTokenizer::new());
        let hf_config = HfTokenizerConfig::default();
        let hf_tokenizer = HfTokenizer::new(word_tokenizer, hf_config);
        
        let encoded = hf_tokenizer.encode("Hello world", true).unwrap();
        assert!(!encoded.input_ids.is_empty());
        assert!(!encoded.tokens.is_empty());
    }
    
    #[test]
    fn test_classification_pipeline() {
        let pipeline = TextClassificationPipeline::new();
        let results = pipeline.predict("This is a great movie!").unwrap();
        
        assert!(!results.is_empty());
        assert!(results[0].score >= 0.0 && results[0].score <= 1.0);
    }
    
    #[test]
    fn test_fill_mask_pipeline() {
        let pipeline = FillMaskPipeline::new();
        let results = pipeline.fill_mask("This is [MASK] example.").unwrap();
        
        assert!(!results.is_empty());
        assert!(results[0].sequence.contains("example"));
    }
    
    #[test]
    fn test_zero_shot_classification() {
        let pipeline = ZeroShotClassificationPipeline::new();
        let labels = vec!["positive", "negative", "neutral"];
        let results = pipeline.classify("This is a great product!", &labels).unwrap();
        
        assert_eq!(results.len(), 3);
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }
    
    #[test]
    fn test_question_answering() {
        let pipeline = QuestionAnsweringPipeline::new();
        let context = "The quick brown fox jumps over the lazy dog.";
        let question = "What jumps over the dog?";
        
        let result = pipeline.answer(question, context).unwrap();
        assert!(!result.answer.is_empty());
        assert!(result.score > 0.0);
        assert!(result.start < result.end);
    }
    
    #[test]
    fn test_hf_model_adapter_pipeline_creation() {
        let config = HfConfig::default();
        let adapter = HfModelAdapter::new(config);
        
        let text_class_pipeline = adapter.create_pipeline("text-classification").unwrap();
        assert!(matches!(text_class_pipeline, HfPipeline::TextClassification(_)));
        
        let zero_shot_pipeline = adapter.create_pipeline("zero-shot-classification").unwrap();
        assert!(matches!(zero_shot_pipeline, HfPipeline::ZeroShotClassification(_)));
        
        let qa_pipeline = adapter.create_pipeline("question-answering").unwrap();
        assert!(matches!(qa_pipeline, HfPipeline::QuestionAnswering(_)));
    }
}