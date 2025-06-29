//! Text processing module for SciRS2
//!
//! This module provides functionality for text processing, tokenization,
//! vectorization, word embeddings, and other NLP-related operations.

#![warn(missing_docs)]

pub mod classification;
pub mod cleansing;
pub mod distance;
pub mod domain_processors;
pub mod embeddings;
pub mod enhanced_vectorize;
pub mod error;
pub mod huggingface_compat;
pub mod information_extraction;
pub mod ml_integration;
pub mod ml_sentiment;
pub mod model_registry;
pub mod multilingual;
pub mod neural_architectures;
pub mod parallel;
pub mod pos_tagging;
pub mod preprocess;
pub mod semantic_similarity;
pub mod sentiment;
pub mod simd_ops;
pub mod sparse;
pub mod sparse_vectorize;
pub mod spelling;
pub mod stemming;
pub mod streaming;
pub mod string_metrics;
pub mod summarization;
pub mod text_statistics;
pub mod token_filter;
pub mod tokenize;
pub mod topic_coherence;
pub mod topic_modeling;
pub mod transformer;
pub mod utils;
pub mod vectorize;
pub mod vocabulary;
pub mod weighted_distance;

// Re-export commonly used items
pub use classification::{
    TextClassificationMetrics, TextClassificationPipeline, TextDataset, TextFeatureSelector,
};
pub use cleansing::{
    expand_contractions, normalize_currencies, normalize_numbers, normalize_ordinals,
    normalize_percentages, normalize_unicode, normalize_whitespace, remove_accents, replace_emails,
    replace_urls, strip_html_tags, AdvancedTextCleaner,
};
pub use distance::{cosine_similarity, jaccard_similarity, levenshtein_distance};
pub use domain_processors::{
    Domain, DomainProcessorConfig, FinancialTextProcessor, LegalTextProcessor, MedicalTextProcessor, 
    ProcessedDomainText, ScientificTextProcessor, UnifiedDomainProcessor,
};
pub use embeddings::{Word2Vec, Word2VecAlgorithm, Word2VecConfig};
pub use enhanced_vectorize::{EnhancedCountVectorizer, EnhancedTfidfVectorizer};
pub use error::{Result, TextError};
pub use huggingface_compat::{
    ClassificationResult, FeatureExtractionPipeline, FillMaskPipeline, FillMaskResult, 
    FormatConverter, HfConfig, HfEncodedInput, HfHub, HfModelAdapter, HfPipeline, 
    HfTokenizer, HfTokenizerConfig, QuestionAnsweringPipeline, QuestionAnsweringResult,
    TextClassificationPipeline, ZeroShotClassificationPipeline,
};
pub use information_extraction::{
    Entity, EntityType, ExtractedInformation, InformationExtractionPipeline, KeyPhraseExtractor,
    PatternExtractor, Relation, RelationExtractor, RuleBasedNER,
};
pub use ml_integration::{
    BatchTextProcessor, FeatureExtractionMode, MLTextPreprocessor, TextFeatures, TextMLPipeline,
};
pub use ml_sentiment::{
    ClassMetrics, EvaluationMetrics, MLSentimentAnalyzer, MLSentimentConfig, TrainingMetrics,
};
pub use model_registry::{
    ModelMetadata, ModelRegistry, ModelType, PrebuiltModels, RegistrableModel, SerializableModelData,
};
pub use multilingual::{
    Language, LanguageDetectionResult, LanguageDetector, MultilingualProcessor, ProcessedText,
    StopWords,
};
pub use neural_architectures::{
    ActivationFunction, AdditiveAttention, BiLSTM, CNNLSTMHybrid, Conv1D, CrossAttention, 
    GRUCell, LSTMCell, MaxPool1D, MultiScaleCNN, PositionwiseFeedForward, ResidualBlock1D, 
    SelfAttention, TextCNN,
};
pub use parallel::{
    ParallelCorpusProcessor, ParallelTextProcessor, ParallelTokenizer, ParallelVectorizer,
};
pub use pos_tagging::{
    PosAwareLemmatizer, PosTagResult, PosTagger, PosTaggerConfig, PosTaggingResult,
};
pub use preprocess::{BasicNormalizer, BasicTextCleaner, TextCleaner, TextNormalizer};
pub use semantic_similarity::{
    LcsSimilarity, SemanticSimilarityEnsemble, SoftCosineSimilarity, WeightedJaccard,
    WordMoversDistance,
};
pub use sentiment::{
    LexiconSentimentAnalyzer, RuleBasedSentimentAnalyzer, Sentiment, SentimentLexicon,
    SentimentResult, SentimentRules, SentimentWordCounts,
};
pub use simd_ops::{SimdEditDistance, SimdStringOps, SimdTextAnalyzer};
pub use sparse::{CsrMatrix, DokMatrix, SparseMatrixBuilder, SparseVector};
pub use sparse_vectorize::{
    sparse_cosine_similarity, MemoryStats, SparseCountVectorizer, SparseTfidfVectorizer,
};
pub use spelling::{
    DictionaryCorrector, DictionaryCorrectorConfig, EditOp, ErrorModel, NGramModel,
    SpellingCorrector, StatisticalCorrector, StatisticalCorrectorConfig,
};
pub use stemming::{
    LancasterStemmer, LemmatizerConfig, PorterStemmer, PosTag, RuleLemmatizer,
    RuleLemmatizerBuilder, SimpleLemmatizer, SnowballStemmer, Stemmer,
};
pub use streaming::{
    ChunkedCorpusReader, MemoryMappedCorpus, ProgressTracker, StreamingTextProcessor,
    StreamingVectorizer,
};
pub use string_metrics::{
    AlignmentResult, DamerauLevenshteinMetric, Metaphone, NeedlemanWunsch, Nysiis,
    PhoneticAlgorithm, SmithWaterman, Soundex, StringMetric,
};
pub use summarization::{CentroidSummarizer, KeywordExtractor, TextRank};
pub use text_statistics::{ReadabilityMetrics, TextMetrics, TextStatistics};
pub use token_filter::{
    CompositeFilter, CustomFilter, FrequencyFilter, LengthFilter, RegexFilter, StopwordsFilter,
    TokenFilter,
};
pub use tokenize::{
    bpe::{BpeConfig, BpeTokenizer, BpeVocabulary},
    CharacterTokenizer, NgramTokenizer, RegexTokenizer, SentenceTokenizer, Tokenizer,
    WhitespaceTokenizer, WordTokenizer,
};
pub use topic_coherence::{TopicCoherence, TopicDiversity};
pub use topic_modeling::{
    LatentDirichletAllocation, LdaBuilder, LdaConfig, LdaLearningMethod, Topic,
};
pub use transformer::{
    FeedForward, LayerNorm, MultiHeadAttention, PositionalEncoding, TokenEmbedding,
    TransformerConfig, TransformerEncoder, TransformerEncoderLayer, TransformerModel,
};
pub use vectorize::{CountVectorizer, TfidfVectorizer, Vectorizer};
pub use vocabulary::Vocabulary;
pub use weighted_distance::{
    DamerauLevenshteinWeights, LevenshteinWeights, WeightedDamerauLevenshtein, WeightedLevenshtein,
    WeightedStringMetric,
};
