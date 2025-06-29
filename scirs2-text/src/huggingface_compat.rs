//! Hugging Face compatibility layer for interoperability
//!
//! This module provides compatibility interfaces and adapters to work with
//! Hugging Face model formats, tokenizers, and APIs, enabling seamless
//! integration with the broader ML ecosystem.

use crate::error::{Result, TextError};
use crate::model_registry::{
    ModelMetadata, ModelRegistry,
};
use crate::tokenize::Tokenizer;
use crate::transformer::{TransformerConfig, TransformerModel};
use ndarray::Array2;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde-support")]
use serde_json;

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
    #[cfg(feature = "serde-support")]
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
            #[cfg(feature = "serde-support")]
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
            #[cfg(feature = "serde-support")]
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
        let mut tokens = self.tokenizer.tokenize(text)?;

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
        let input_ids: Vec<usize> = tokens
            .iter()
            .map(|token| {
                self.vocab
                    .get(token)
                    .copied()
                    .unwrap_or(self.vocab.get(&self.config.unk_token).copied().unwrap_or(0))
            })
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
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<HfEncodedInput>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> Result<String> {
        let tokens: Vec<String> = token_ids
            .iter()
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
    pub fn load_from_hf_directory<P: AsRef<Path>>(
        &self,
        model_path: P,
    ) -> Result<TransformerModel> {
        let model_path = model_path.as_ref();

        // Check for required files
        let config_file = model_path.join("config.json");
        if !config_file.exists() {
            return Err(TextError::InvalidInput(
                "HF config.json not found".to_string(),
            ));
        }

        // Load configuration
        let transformer_config = if config_file.exists() {
            #[cfg(feature = "serde-support")]
            {
                let file = fs::File::open(&config_file).map_err(|e| {
                    TextError::IoError(format!("Failed to open config file: {}", e))
                })?;
                let reader = BufReader::new(file);
                let hf_config: HfConfig = serde_json::from_reader(reader).map_err(|e| {
                    TextError::InvalidInput(format!("Failed to deserialize config: {}", e))
                })?;
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
        _model: &TransformerModel,
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
            serde_json::to_writer_pretty(writer, &self.config).map_err(|e| {
                TextError::InvalidInput(format!("Failed to serialize config: {}", e))
            })?;
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
            let tokenizer_file =
                fs::File::create(output_path.join("tokenizer.json")).map_err(|e| {
                    TextError::IoError(format!("Failed to create tokenizer file: {}", e))
                })?;
            let writer = BufWriter::new(tokenizer_file);
            serde_json::to_writer_pretty(writer, &tokenizer_config).map_err(|e| {
                TextError::InvalidInput(format!("Failed to serialize tokenizer config: {}", e))
            })?;
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
                TextClassificationPipeline::new(),
            )),
            "feature-extraction" => Ok(HfPipeline::FeatureExtraction(
                FeatureExtractionPipeline::new(),
            )),
            "fill-mask" => Ok(HfPipeline::FillMask(FillMaskPipeline::new())),
            "zero-shot-classification" => Ok(HfPipeline::ZeroShotClassification(
                ZeroShotClassificationPipeline::new(),
            )),
            "question-answering" => Ok(HfPipeline::QuestionAnswering(
                QuestionAnsweringPipeline::new(),
            )),
            "text-generation" => Ok(HfPipeline::TextGeneration(TextGenerationPipeline::new())),
            "summarization" => Ok(HfPipeline::Summarization(SummarizationPipeline::new())),
            "translation" => Ok(HfPipeline::Translation(TranslationPipeline::new())),
            "token-classification" => Ok(HfPipeline::TokenClassification(
                TokenClassificationPipeline::new(),
            )),
            _ => Err(TextError::InvalidInput(format!(
                "Unsupported task: {}",
                task
            ))),
        }
    }

    /// Create pipeline from model directory
    pub fn create_pipeline_from_model<P: AsRef<Path>>(
        &self,
        model_path: P,
        task: Option<&str>,
    ) -> Result<HfPipeline> {
        let model_path = model_path.as_ref();

        // Load config to infer task if not provided
        let config_file = model_path.join("config.json");
        let inferred_task = if config_file.exists() && task.is_none() {
            // Try to infer task from config
            "text-classification" // Default fallback
        } else {
            task.unwrap_or("text-classification")
        };

        self.create_pipeline(inferred_task)
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
    /// Text generation pipeline
    TextGeneration(TextGenerationPipeline),
    /// Summarization pipeline
    Summarization(SummarizationPipeline),
    /// Translation pipeline
    Translation(TranslationPipeline),
    /// Token classification pipeline
    TokenClassification(TokenClassificationPipeline),
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
    pub fn classify(
        &self,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<ClassificationResult>> {
        let mut results = Vec::new();

        // Simplified zero-shot classification (would use actual NLI model)
        for (_i, &label) in candidate_labels.iter().enumerate() {
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
pub struct HfHub {
    /// Base URL for HF Hub API
    api_base: String,
    /// User token for authentication
    token: Option<String>,
}

impl HfHub {
    /// Create new HF Hub client
    pub fn new() -> Self {
        Self {
            api_base: "https://huggingface.co".to_string(),
            token: None,
        }
    }

    /// Set authentication token
    pub fn with_token(mut self, token: String) -> Self {
        self.token = Some(token);
        self
    }

    /// Download model from Hugging Face Hub
    pub fn download_model(&self, model_id: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
        // Use environment variable or default path
        let default_cache = if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home)
                .join(".cache")
                .join("huggingface")
                .join("hub")
        } else {
            PathBuf::from(".")
                .join(".cache")
                .join("huggingface")
                .join("hub")
        };

        let cache_dir = cache_dir.unwrap_or(&default_cache);
        let model_path = cache_dir.join(model_id.replace("/", "--"));

        if !model_path.exists() {
            std::fs::create_dir_all(&model_path)
                .map_err(|e| TextError::IoError(format!("Failed to create cache dir: {}", e)))?;

            // Download model files
            self.download_model_files(model_id, &model_path)?;
        }

        Ok(model_path)
    }

    /// Download model files
    fn download_model_files(&self, model_id: &str, cache_path: &Path) -> Result<()> {
        // Essential files to download
        let files_to_download = vec![
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "pytorch_model.bin",
            "model.safetensors",
        ];

        for file in files_to_download {
            if let Ok(content) = self.download_file(model_id, file) {
                let file_path = cache_path.join(file);
                std::fs::write(&file_path, content)
                    .map_err(|e| TextError::IoError(format!("Failed to write {}: {}", file, e)))?;
            }
        }

        Ok(())
    }

    /// Download a specific file from model repository
    fn download_file(&self, _model_id: &str, filename: &str) -> Result<Vec<u8>> {
        // Create mock file content for demonstration
        // In a real implementation, this would make HTTP requests to HF Hub
        let mock_content = match filename {
            "config.json" => {
                #[cfg(feature = "serde-support")]
                {
                    let config = HfConfig::default();
                    serde_json::to_string_pretty(&config)
                        .map_err(|e| TextError::InvalidInput(format!("JSON error: {}", e)))?
                        .into_bytes()
                }
                #[cfg(not(feature = "serde-support"))]
                {
                    // Fallback JSON without serde
                    format!(
                        r#"{{
    "architectures": ["BertModel"],
    "model_type": "bert",
    "num_attention_heads": 12,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "vocab_size": 30522,
    "max_position_embeddings": 512
}}"#
                    )
                    .into_bytes()
                }
            }
            "tokenizer_config.json" => {
                #[cfg(feature = "serde-support")]
                {
                    let tokenizer_config = HfTokenizerConfig::default();
                    serde_json::to_string_pretty(&tokenizer_config)
                        .map_err(|e| TextError::InvalidInput(format!("JSON error: {}", e)))?
                        .into_bytes()
                }
                #[cfg(not(feature = "serde-support"))]
                {
                    // Fallback JSON without serde
                    format!(
                        r#"{{
    "tokenizer_type": "WordPiece",
    "max_len": 512,
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "bos_token": "[CLS]",
    "eos_token": "[SEP]"
}}"#
                    )
                    .into_bytes()
                }
            }
            "vocab.txt" => {
                // Mock vocabulary
                (0..1000)
                    .map(|i| format!("[TOKEN_{}]", i))
                    .collect::<Vec<_>>()
                    .join("\n")
                    .into_bytes()
            }
            _ => {
                // Mock binary data
                vec![0u8; 1024]
            }
        };

        Ok(mock_content)
    }

    /// List available models with filtering
    pub fn list_models(
        &self,
        filter_task: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<HfModelInfo>> {
        let mut models = vec![
            HfModelInfo {
                id: "bert-base-uncased".to_string(),
                author: "google".to_string(),
                pipeline_tag: Some("fill-mask".to_string()),
                tags: vec!["pytorch".to_string(), "bert".to_string()],
                downloads: 1000000,
                likes: 500,
                library_name: Some("transformers".to_string()),
                created_at: "2020-01-01T00:00:00Z".to_string(),
            },
            HfModelInfo {
                id: "roberta-base".to_string(),
                author: "facebook".to_string(),
                pipeline_tag: Some("fill-mask".to_string()),
                tags: vec!["pytorch".to_string(), "roberta".to_string()],
                downloads: 800000,
                likes: 400,
                library_name: Some("transformers".to_string()),
                created_at: "2020-02-01T00:00:00Z".to_string(),
            },
            HfModelInfo {
                id: "distilbert-base-uncased".to_string(),
                author: "huggingface".to_string(),
                pipeline_tag: Some("text-classification".to_string()),
                tags: vec!["pytorch".to_string(), "distilbert".to_string()],
                downloads: 900000,
                likes: 600,
                library_name: Some("transformers".to_string()),
                created_at: "2020-03-01T00:00:00Z".to_string(),
            },
            HfModelInfo {
                id: "gpt2".to_string(),
                author: "openai".to_string(),
                pipeline_tag: Some("text-generation".to_string()),
                tags: vec!["pytorch".to_string(), "gpt2".to_string()],
                downloads: 1200000,
                likes: 800,
                library_name: Some("transformers".to_string()),
                created_at: "2019-11-01T00:00:00Z".to_string(),
            },
        ];

        // Filter by task if specified
        if let Some(task) = filter_task {
            models.retain(|model| {
                model
                    .pipeline_tag
                    .as_ref()
                    .map_or(false, |tag| tag.contains(task))
            });
        }

        // Apply limit
        if let Some(limit) = limit {
            models.truncate(limit);
        }

        Ok(models)
    }

    /// Get detailed model information
    pub fn model_info(&self, model_id: &str) -> Result<HfModelInfo> {
        // In a real implementation, this would fetch from HF API
        let models = self.list_models(None, None)?;
        models
            .into_iter()
            .find(|model| model.id == model_id)
            .ok_or_else(|| TextError::InvalidInput(format!("Model not found: {}", model_id)))
    }

    /// Search models by query
    pub fn search_models(&self, query: &str, limit: Option<usize>) -> Result<Vec<HfModelInfo>> {
        let models = self.list_models(None, None)?;
        let mut filtered: Vec<_> = models
            .into_iter()
            .filter(|model| {
                model.id.to_lowercase().contains(&query.to_lowercase())
                    || model
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query.to_lowercase()))
            })
            .collect();

        if let Some(limit) = limit {
            filtered.truncate(limit);
        }

        Ok(filtered)
    }

    /// Get trending models
    pub fn trending_models(&self, _period: &str, limit: Option<usize>) -> Result<Vec<HfModelInfo>> {
        let mut models = self.list_models(None, None)?;

        // Sort by downloads (as a proxy for trending)
        models.sort_by(|a, b| b.downloads.cmp(&a.downloads));

        if let Some(limit) = limit {
            models.truncate(limit);
        }

        Ok(models)
    }
}

impl Default for HfHub {
    fn default() -> Self {
        Self::new()
    }
}

/// Model information from HF Hub
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct HfModelInfo {
    /// Model identifier
    pub id: String,
    /// Model author/organization
    pub author: String,
    /// Pipeline task tag
    pub pipeline_tag: Option<String>,
    /// Model tags
    pub tags: Vec<String>,
    /// Number of downloads
    pub downloads: u64,
    /// Number of likes
    pub likes: u64,
    /// Library name (e.g., "transformers")
    pub library_name: Option<String>,
    /// Creation timestamp
    pub created_at: String,
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

/// Text generation pipeline
#[derive(Debug)]
pub struct TextGenerationPipeline {
    max_length: usize,
    temperature: f64,
}

impl TextGenerationPipeline {
    /// Create new text generation pipeline
    pub fn new() -> Self {
        Self {
            max_length: 50,
            temperature: 1.0,
        }
    }

    /// Generate text continuation
    pub fn generate(&self, prompt: &str) -> Result<Vec<TextGenerationResult>> {
        // Simplified text generation (would use actual language model)
        let words = vec![
            "the", "and", "a", "to", "of", "in", "for", "with", "on", "by",
        ];
        let mut generated = prompt.to_string();

        for _ in 0..10 {
            use rand::Rng;
            let mut rng = rand::rng();
            if let Some(word) = words.get(rng.random_range(0..words.len())) {
                generated.push_str(" ");
                generated.push_str(word);
            }
        }

        Ok(vec![TextGenerationResult {
            generated_text: generated,
            score: 0.8,
        }])
    }
}

/// Text generation result
#[derive(Debug, Clone)]
pub struct TextGenerationResult {
    /// Generated text
    pub generated_text: String,
    /// Generation score
    pub score: f64,
}

/// Summarization pipeline
#[derive(Debug)]
pub struct SummarizationPipeline {
    max_length: usize,
    min_length: usize,
}

impl SummarizationPipeline {
    /// Create new summarization pipeline
    pub fn new() -> Self {
        Self {
            max_length: 100,
            min_length: 10,
        }
    }

    /// Summarize text
    pub fn summarize(&self, text: &str) -> Result<SummarizationResult> {
        // Simplified summarization (extractive approach)
        let sentences: Vec<&str> = text.split('.').collect();
        let summary = if sentences.len() > 2 {
            format!("{}. {}.", sentences[0], sentences[1])
        } else {
            text.to_string()
        };

        Ok(SummarizationResult {
            summary_text: summary,
            score: 0.7,
        })
    }
}

/// Summarization result
#[derive(Debug, Clone)]
pub struct SummarizationResult {
    /// Summary text
    pub summary_text: String,
    /// Summarization score
    pub score: f64,
}

/// Translation pipeline
#[derive(Debug)]
pub struct TranslationPipeline {
    source_lang: String,
    target_lang: String,
}

impl TranslationPipeline {
    /// Create new translation pipeline
    pub fn new() -> Self {
        Self {
            source_lang: "en".to_string(),
            target_lang: "fr".to_string(),
        }
    }

    /// Set source and target languages
    pub fn with_languages(mut self, source: String, target: String) -> Self {
        self.source_lang = source;
        self.target_lang = target;
        self
    }

    /// Translate text
    pub fn translate(&self, text: &str) -> Result<TranslationResult> {
        // Mock translation (would use actual translation model)
        let mock_translations = [
            ("hello", "bonjour"),
            ("world", "monde"),
            ("good", "bon"),
            ("morning", "matin"),
            ("thank you", "merci"),
        ];

        let mut translated = text.to_lowercase();
        for (en, fr) in &mock_translations {
            translated = translated.replace(en, fr);
        }

        Ok(TranslationResult {
            translation_text: translated,
            score: 0.9,
        })
    }
}

/// Translation result
#[derive(Debug, Clone)]
pub struct TranslationResult {
    /// Translated text
    pub translation_text: String,
    /// Translation score
    pub score: f64,
}

/// Token classification pipeline (NER, POS tagging, etc.)
#[derive(Debug)]
pub struct TokenClassificationPipeline {
    aggregation_strategy: String,
}

impl TokenClassificationPipeline {
    /// Create new token classification pipeline
    pub fn new() -> Self {
        Self {
            aggregation_strategy: "simple".to_string(),
        }
    }

    /// Classify tokens in text
    pub fn classify_tokens(&self, text: &str) -> Result<Vec<TokenClassificationResult>> {
        // Simplified NER (would use actual NER model)
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut results = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let (entity, score) = if word.chars().next().map_or(false, |c| c.is_uppercase()) {
                ("PERSON", 0.9)
            } else if word.contains("@") {
                ("EMAIL", 0.95)
            } else if word.parse::<f64>().is_ok() {
                ("NUMBER", 0.85)
            } else {
                ("O", 0.1) // Outside any entity
            };

            if entity != "O" {
                results.push(TokenClassificationResult {
                    entity: entity.to_string(),
                    score,
                    index: i,
                    word: word.to_string(),
                    start: 0, // Simplified
                    end: word.len(),
                });
            }
        }

        Ok(results)
    }
}

/// Token classification result
#[derive(Debug, Clone)]
pub struct TokenClassificationResult {
    /// Entity type
    pub entity: String,
    /// Confidence score
    pub score: f64,
    /// Token index
    pub index: usize,
    /// Token word
    pub word: String,
    /// Start character position
    pub start: usize,
    /// End character position
    pub end: usize,
}

/// Model manager for HF compatibility
pub struct HfModelManager {
    hub: HfHub,
    registry: Option<ModelRegistry>,
}

impl HfModelManager {
    /// Create new model manager
    pub fn new() -> Self {
        Self {
            hub: HfHub::new(),
            registry: None,
        }
    }

    /// Set model registry
    pub fn with_registry(mut self, registry: ModelRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Load model from HF Hub or local cache
    pub fn load_model(&self, model_id: &str, cache_dir: Option<&Path>) -> Result<TransformerModel> {
        // First try to download from HF Hub
        let model_path = self.hub.download_model(model_id, cache_dir)?;

        // Create adapter and load model
        let adapter = HfModelAdapter::new(HfConfig::default());
        adapter.load_from_hf_directory(&model_path)
    }

    /// Save model in HF format
    pub fn save_model<P: AsRef<Path>>(
        &self,
        model: &TransformerModel,
        output_path: P,
        model_id: &str,
    ) -> Result<()> {
        let config = HfConfig::from_transformer_config(&model.config);
        let adapter = HfModelAdapter::new(config);
        adapter.save_to_hf_directory(model, output_path)
    }

    /// Convert SciRS2 model to HF format
    pub fn convert_to_hf(&self, model: &TransformerModel) -> Result<HfConfig> {
        Ok(HfConfig::from_transformer_config(&model.config))
    }

    /// Get available models
    pub fn list_available_models(&self, task: Option<&str>) -> Result<Vec<HfModelInfo>> {
        self.hub.list_models(task, None)
    }
}

impl Default for HfModelManager {
    fn default() -> Self {
        Self::new()
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
        let results = pipeline
            .classify("This is a great product!", &labels)
            .unwrap();

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
        assert!(matches!(
            text_class_pipeline,
            HfPipeline::TextClassification(_)
        ));

        let zero_shot_pipeline = adapter.create_pipeline("zero-shot-classification").unwrap();
        assert!(matches!(
            zero_shot_pipeline,
            HfPipeline::ZeroShotClassification(_)
        ));

        let qa_pipeline = adapter.create_pipeline("question-answering").unwrap();
        assert!(matches!(qa_pipeline, HfPipeline::QuestionAnswering(_)));
    }
}
