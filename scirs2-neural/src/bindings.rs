//! C/C++ binding generation utilities for neural networks
//!
//! This module provides comprehensive tools for generating C and C++ bindings including:
//! - Automatic header generation with proper type mappings
//! - Source code generation for implementation stubs
//! - Build system integration (CMake, Makefile)
//! - Cross-platform compatibility handling
//! - Advanced binding features (callbacks, memory management)

use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
use crate::serving::{CallingConvention, PackageMetadata, TensorSpec};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// C/C++ binding generation configuration
#[derive(Debug, Clone)]
pub struct BindingConfig {
    /// Library name
    pub library_name: String,
    /// Target language (C or C++)
    pub language: BindingLanguage,
    /// API style configuration
    pub api_style: ApiStyle,
    /// Type mapping configuration
    pub type_mappings: TypeMappings,
    /// Memory management strategy
    pub memory_strategy: MemoryStrategy,
    /// Error handling approach
    pub error_handling: ErrorHandling,
    /// Threading configuration
    pub threading: ThreadingConfig,
    /// Build system configuration
    pub build_system: BuildSystemConfig,
}

/// Target binding language
#[derive(Debug, Clone, PartialEq)]
pub enum BindingLanguage {
    /// Pure C bindings
    C,
    /// C++ bindings with classes
    Cpp,
    /// C bindings with C++ wrapper
    CWithCppWrapper,
}

/// API style for bindings
#[derive(Debug, Clone, PartialEq)]
pub enum ApiStyle {
    /// Procedural API (function-based)
    Procedural,
    /// Object-oriented API (class-based)
    ObjectOriented,
    /// Hybrid approach
    Hybrid,
}

/// Type mapping configuration
#[derive(Debug, Clone)]
pub struct TypeMappings {
    /// Primitive type mappings
    pub primitives: HashMap<String, String>,
    /// Array type mappings
    pub arrays: ArrayMapping,
    /// String handling
    pub strings: StringMapping,
    /// Custom type definitions
    pub custom_types: Vec<CustomType>,
}

/// Array type mapping strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayMapping {
    /// Use plain C arrays with separate size parameter
    PlainArrays,
    /// Use structure with data pointer and metadata
    StructuredArrays,
    /// Use custom array type
    CustomArrayType(String),
}

/// String handling strategy
#[derive(Debug, Clone, PartialEq)]
pub enum StringMapping {
    /// Use null-terminated C strings
    CString,
    /// Use length-prefixed strings
    LengthPrefixed,
    /// Use custom string type
    CustomString(String),
}

/// Custom type definition
#[derive(Debug, Clone)]
pub struct CustomType {
    /// Type name in Rust
    pub rust_name: String,
    /// Type name in C/C++
    pub c_name: String,
    /// Type definition
    pub definition: String,
    /// Include dependencies
    pub includes: Vec<String>,
}

/// Memory management strategy
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryStrategy {
    /// Manual memory management
    Manual,
    /// Reference counting
    ReferenceCounting,
    /// RAII with smart pointers (C++ only)
    RAII,
    /// Custom allocator
    CustomAllocator(String),
}

/// Error handling approach
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorHandling {
    /// Return error codes
    ErrorCodes,
    /// Use errno
    Errno,
    /// C++ exceptions (C++ only)
    Exceptions,
    /// Callback-based error handling
    Callbacks,
}

/// Threading configuration
#[derive(Debug, Clone)]
pub struct ThreadingConfig {
    /// Thread safety level
    pub safety_level: ThreadSafety,
    /// Synchronization primitives
    pub sync_primitives: Vec<SyncPrimitive>,
    /// Thread pool configuration
    pub thread_pool: Option<ThreadPoolConfig>,
}

/// Thread safety level
#[derive(Debug, Clone, PartialEq)]
pub enum ThreadSafety {
    /// Not thread-safe
    None,
    /// Thread-safe for reads
    ReadOnly,
    /// Fully thread-safe
    Full,
    /// Thread-local storage
    ThreadLocal,
}

/// Synchronization primitive
#[derive(Debug, Clone, PartialEq)]
pub enum SyncPrimitive {
    /// Mutex
    Mutex,
    /// Read-write lock
    RwLock,
    /// Atomic operations
    Atomic,
    /// Condition variables
    CondVar,
}

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Default thread count
    pub default_threads: Option<usize>,
    /// Minimum threads
    pub min_threads: usize,
    /// Maximum threads
    pub max_threads: usize,
    /// Thread naming pattern
    pub thread_name_pattern: String,
}

/// Build system configuration
#[derive(Debug, Clone)]
pub struct BuildSystemConfig {
    /// Target build system
    pub system: BuildSystem,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
    /// Linker flags
    pub linker_flags: Vec<String>,
    /// Dependencies
    pub dependencies: Vec<Dependency>,
    /// Install configuration
    pub install_config: InstallConfig,
}

/// Build system type
#[derive(Debug, Clone, PartialEq)]
pub enum BuildSystem {
    /// CMake
    CMake,
    /// GNU Make
    Make,
    /// Meson
    Meson,
    /// Bazel
    Bazel,
    /// Custom build system
    Custom(String),
}

/// External dependency
#[derive(Debug, Clone)]
pub struct Dependency {
    /// Dependency name
    pub name: String,
    /// Version requirement
    pub version: Option<String>,
    /// Required headers
    pub headers: Vec<String>,
    /// Required libraries
    pub libraries: Vec<String>,
    /// Package manager (pkg-config, vcpkg, etc.)
    pub package_manager: Option<String>,
}

/// Installation configuration
#[derive(Debug, Clone)]
pub struct InstallConfig {
    /// Installation prefix
    pub prefix: String,
    /// Binary directory
    pub bin_dir: String,
    /// Library directory
    pub lib_dir: String,
    /// Include directory
    pub include_dir: String,
    /// Generate pkg-config file
    pub generate_pkgconfig: bool,
}

/// Generated binding result
#[derive(Debug, Clone)]
pub struct BindingResult {
    /// Generated header files
    pub headers: Vec<PathBuf>,
    /// Generated source files
    pub sources: Vec<PathBuf>,
    /// Build system files
    pub build_files: Vec<PathBuf>,
    /// Example files
    pub examples: Vec<PathBuf>,
    /// Documentation files
    pub documentation: Vec<PathBuf>,
}

/// C/C++ binding generator
pub struct BindingGenerator<F: Float + Debug + ndarray::ScalarOperand> {
    /// Model to generate bindings for
    model: Sequential<F>,
    /// Binding configuration
    config: BindingConfig,
    /// Package metadata
    metadata: PackageMetadata,
    /// Output directory
    output_dir: PathBuf,
}

impl<F: Float + Debug + 'static + num_traits::FromPrimitive + ndarray::ScalarOperand + Send + Sync> BindingGenerator<F> {
    /// Create a new binding generator
    pub fn new(
        model: Sequential<F>,
        config: BindingConfig,
        metadata: PackageMetadata,
        output_dir: PathBuf,
    ) -> Self {
        Self {
            model,
            config,
            metadata,
            output_dir,
        }
    }

    /// Generate complete bindings
    pub fn generate(&self) -> Result<BindingResult> {
        // Create output directory structure
        self.create_directory_structure()?;

        let mut result = BindingResult {
            headers: Vec::new(),
            sources: Vec::new(),
            build_files: Vec::new(),
            examples: Vec::new(),
            documentation: Vec::new(),
        };

        // Generate header files
        let header_path = self.generate_header()?;
        result.headers.push(header_path);

        // Generate source files
        let source_path = self.generate_source()?;
        result.sources.push(source_path);

        // Generate C++ wrapper if needed
        if self.config.language == BindingLanguage::CWithCppWrapper {
            let (cpp_header, cpp_source) = self.generate_cpp_wrapper()?;
            result.headers.push(cpp_header);
            result.sources.push(cpp_source);
        }

        // Generate build system files
        let build_files = self.generate_build_system()?;
        result.build_files.extend(build_files);

        // Generate examples
        let example_files = self.generate_examples()?;
        result.examples.extend(example_files);

        // Generate documentation
        let doc_files = self.generate_documentation()?;
        result.documentation.extend(doc_files);

        Ok(result)
    }

    fn create_directory_structure(&self) -> Result<()> {
        let dirs = vec!["include", "src", "examples", "docs", "build"];
        for dir in dirs {
            let path = self.output_dir.join(dir);
            fs::create_dir_all(&path)
                .map_err(|e| NeuralError::IOError(format!("Failed to create directory {}: {}", path.display(), e)))?;
        }
        Ok(())
    }

    fn generate_header(&self) -> Result<PathBuf> {
        let header_path = self.output_dir.join("include").join(format!("{}.h", self.config.library_name));
        
        let header_guard = format!("{}_H", self.config.library_name.to_uppercase());
        let mut header_content = String::new();

        // Header guard and includes
        header_content.push_str(&format!(
            r#"#ifndef {}
#define {}

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

"#, header_guard, header_guard));

        // Add custom includes
        for custom_type in &self.config.type_mappings.custom_types {
            for include in &custom_type.includes {
                header_content.push_str(&format!("#include <{}>\n", include));
            }
        }

        // C++ compatibility
        header_content.push_str(r#"
#ifdef __cplusplus
extern "C" {
#endif

"#);

        // Generate type definitions
        header_content.push_str(&self.generate_type_definitions()?);

        // Generate API declarations
        header_content.push_str(&self.generate_api_declarations()?);

        // Close C++ compatibility
        header_content.push_str(r#"
#ifdef __cplusplus
}
#endif

"#);

        // Close header guard
        header_content.push_str(&format!("#endif // {}\n", header_guard));

        fs::write(&header_path, header_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(header_path)
    }

    fn generate_type_definitions(&self) -> Result<String> {
        let mut defs = String::new();

        // Generate tensor type based on array mapping
        match self.config.type_mappings.arrays {
            ArrayMapping::PlainArrays => {
                defs.push_str(r#"
// Plain array interface
typedef struct {
    void* data;
    size_t element_size;
    size_t* shape;
    size_t ndim;
    size_t total_elements;
} scirs2_tensor_t;

"#);
            }
            ArrayMapping::StructuredArrays => {
                defs.push_str(r#"
// Structured array interface
typedef struct {
    void* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    int dtype;
    bool owns_data;
} scirs2_tensor_t;

"#);
            }
            ArrayMapping::CustomArrayType(ref type_name) => {
                defs.push_str(&format!(r#"
// Custom array type
typedef {} scirs2_tensor_t;

"#, type_name));
            }
        }

        // Generate model handle type
        defs.push_str(r#"
// Model handle
typedef struct {
    void* internal_handle;
    bool is_valid;
} scirs2_model_t;

"#);

        // Generate error type based on error handling strategy
        match self.config.error_handling {
            ErrorHandling::ErrorCodes => {
                defs.push_str(r#"
// Error codes
typedef enum {
    SCIRS2_SUCCESS = 0,
    SCIRS2_ERROR_NULL_POINTER = -1,
    SCIRS2_ERROR_INVALID_MODEL = -2,
    SCIRS2_ERROR_INVALID_TENSOR = -3,
    SCIRS2_ERROR_DIMENSION_MISMATCH = -4,
    SCIRS2_ERROR_OUT_OF_MEMORY = -5,
    SCIRS2_ERROR_IO_ERROR = -6,
    SCIRS2_ERROR_COMPUTATION_ERROR = -7,
    SCIRS2_ERROR_UNKNOWN = -99
} scirs2_error_t;

"#);
            }
            ErrorHandling::Callbacks => {
                defs.push_str(r#"
// Error callback type
typedef void (*scirs2_error_callback_t)(int error_code, const char* message, void* user_data);

// Error handling context
typedef struct {
    scirs2_error_callback_t callback;
    void* user_data;
} scirs2_error_context_t;

"#);
            }
            _ => {}
        }

        // Generate custom types
        for custom_type in &self.config.type_mappings.custom_types {
            defs.push_str(&format!("// Custom type: {}\n", custom_type.rust_name));
            defs.push_str(&custom_type.definition);
            defs.push_str("\n\n");
        }

        Ok(defs)
    }

    fn generate_api_declarations(&self) -> Result<String> {
        let mut api = String::new();

        // Add function attributes based on calling convention
        let call_conv = match self.config.api_style {
            ApiStyle::Procedural => self.generate_procedural_api()?,
            ApiStyle::ObjectOriented => self.generate_oo_api()?,
            ApiStyle::Hybrid => self.generate_hybrid_api()?,
        };

        api.push_str(&call_conv);

        Ok(api)
    }

    fn generate_procedural_api(&self) -> Result<String> {
        let mut api = String::new();

        api.push_str(r#"
// === Core Model Functions ===

/**
 * Load a model from file
 * @param model_path Path to the model file
 * @param model Output model handle
 * @return Error code (0 = success)
 */
scirs2_error_t scirs2_model_load(const char* model_path, scirs2_model_t* model);

/**
 * Load a model from memory
 * @param data Model data buffer
 * @param size Size of data buffer
 * @param model Output model handle
 * @return Error code (0 = success)
 */
scirs2_error_t scirs2_model_load_from_memory(const void* data, size_t size, scirs2_model_t* model);

/**
 * Free model resources
 * @param model Model handle to free
 */
void scirs2_model_free(scirs2_model_t* model);

/**
 * Check if model is valid
 * @param model Model handle
 * @return true if valid, false otherwise
 */
bool scirs2_model_is_valid(const scirs2_model_t* model);

// === Inference Functions ===

/**
 * Run inference on input tensor
 * @param model Model handle
 * @param input Input tensor
 * @param output Output tensor (allocated by caller)
 * @return Error code (0 = success)
 */
scirs2_error_t scirs2_model_predict(const scirs2_model_t* model, 
                                   const scirs2_tensor_t* input, 
                                   scirs2_tensor_t* output);

/**
 * Run batch inference
 * @param model Model handle
 * @param inputs Array of input tensors
 * @param batch_size Number of inputs
 * @param outputs Array of output tensors (allocated by caller)
 * @return Error code (0 = success)
 */
scirs2_error_t scirs2_model_predict_batch(const scirs2_model_t* model,
                                         const scirs2_tensor_t* inputs,
                                         size_t batch_size,
                                         scirs2_tensor_t* outputs);

// === Tensor Functions ===

/**
 * Create a new tensor
 * @param shape Tensor shape array
 * @param ndim Number of dimensions
 * @param dtype Data type
 * @param tensor Output tensor
 * @return Error code (0 = success)
 */
scirs2_error_t scirs2_tensor_create(const size_t* shape, size_t ndim, int dtype, scirs2_tensor_t* tensor);

/**
 * Create tensor from data
 * @param data Data buffer
 * @param shape Tensor shape
 * @param ndim Number of dimensions
 * @param dtype Data type
 * @param copy Whether to copy data (true) or take ownership (false)
 * @param tensor Output tensor
 * @return Error code (0 = success)
 */
scirs2_error_t scirs2_tensor_from_data(const void* data, const size_t* shape, 
                                      size_t ndim, int dtype, bool copy, 
                                      scirs2_tensor_t* tensor);

/**
 * Free tensor resources
 * @param tensor Tensor to free
 */
void scirs2_tensor_free(scirs2_tensor_t* tensor);

/**
 * Get tensor data pointer
 * @param tensor Tensor handle
 * @return Data pointer
 */
void* scirs2_tensor_data(const scirs2_tensor_t* tensor);

/**
 * Get tensor shape
 * @param tensor Tensor handle
 * @param shape Output shape array (caller allocated)
 * @param ndim Output number of dimensions
 * @return Error code (0 = success)
 */
scirs2_error_t scirs2_tensor_shape(const scirs2_tensor_t* tensor, size_t* shape, size_t* ndim);

// === Utility Functions ===

/**
 * Get library version
 * @return Version string
 */
const char* scirs2_version(void);

/**
 * Get error message for error code
 * @param error_code Error code
 * @return Error message string
 */
const char* scirs2_error_message(scirs2_error_t error_code);

/**
 * Set global error callback
 * @param callback Error callback function
 * @param user_data User data for callback
 */
void scirs2_set_error_callback(scirs2_error_callback_t callback, void* user_data);

"#);

        // Add threading functions if threading is enabled
        if self.config.threading.safety_level != ThreadSafety::None {
            api.push_str(r#"
// === Threading Functions ===

/**
 * Set number of threads for inference
 * @param num_threads Number of threads (0 = auto)
 * @return Error code (0 = success)
 */
scirs2_error_t scirs2_set_num_threads(size_t num_threads);

/**
 * Get current number of threads
 * @return Number of threads
 */
size_t scirs2_get_num_threads(void);

"#);
        }

        Ok(api)
    }

    fn generate_oo_api(&self) -> Result<String> {
        let mut api = String::new();

        // Object-oriented API is primarily for C++
        if self.config.language == BindingLanguage::Cpp {
            api.push_str(r#"
// === C++ Object-Oriented API ===

class SciRS2Model {
public:
    /**
     * Constructor - load model from file
     * @param model_path Path to model file
     */
    explicit SciRS2Model(const std::string& model_path);
    
    /**
     * Constructor - load model from memory
     * @param data Model data
     * @param size Data size
     */
    SciRS2Model(const void* data, size_t size);
    
    /**
     * Destructor
     */
    ~SciRS2Model();
    
    /**
     * Move constructor
     */
    SciRS2Model(SciRS2Model&& other) noexcept;
    
    /**
     * Move assignment
     */
    SciRS2Model& operator=(SciRS2Model&& other) noexcept;
    
    // Delete copy constructor and assignment
    SciRS2Model(const SciRS2Model&) = delete;
    SciRS2Model& operator=(const SciRS2Model&) = delete;
    
    /**
     * Check if model is valid
     * @return true if valid
     */
    bool is_valid() const;
    
    /**
     * Run inference
     * @param input Input tensor
     * @return Output tensor
     */
    SciRS2Tensor predict(const SciRS2Tensor& input);
    
    /**
     * Run batch inference
     * @param inputs Input tensors
     * @return Output tensors
     */
    std::vector<SciRS2Tensor> predict_batch(const std::vector<SciRS2Tensor>& inputs);
    
    /**
     * Get model metadata
     * @return Model metadata
     */
    const ModelMetadata& get_metadata() const;

private:
    scirs2_model_t model_;
    std::unique_ptr<ModelMetadata> metadata_;
};

class SciRS2Tensor {
public:
    /**
     * Constructor - create from shape
     * @param shape Tensor shape
     * @param dtype Data type
     */
    SciRS2Tensor(const std::vector<size_t>& shape, DataType dtype);
    
    /**
     * Constructor - create from data
     * @param data Data pointer
     * @param shape Tensor shape
     * @param dtype Data type
     * @param copy Whether to copy data
     */
    SciRS2Tensor(const void* data, const std::vector<size_t>& shape, 
                DataType dtype, bool copy = true);
    
    /**
     * Destructor
     */
    ~SciRS2Tensor();
    
    /**
     * Move constructor
     */
    SciRS2Tensor(SciRS2Tensor&& other) noexcept;
    
    /**
     * Move assignment
     */
    SciRS2Tensor& operator=(SciRS2Tensor&& other) noexcept;
    
    // Delete copy constructor and assignment
    SciRS2Tensor(const SciRS2Tensor&) = delete;
    SciRS2Tensor& operator=(const SciRS2Tensor&) = delete;
    
    /**
     * Get tensor data
     * @return Data pointer
     */
    void* data();
    const void* data() const;
    
    /**
     * Get tensor shape
     * @return Shape vector
     */
    const std::vector<size_t>& shape() const;
    
    /**
     * Get number of dimensions
     * @return Number of dimensions
     */
    size_t ndim() const;
    
    /**
     * Get data type
     * @return Data type
     */
    DataType dtype() const;
    
    /**
     * Get total number of elements
     * @return Element count
     */
    size_t size() const;

private:
    scirs2_tensor_t tensor_;
    std::vector<size_t> shape_;
    DataType dtype_;
};

"#);
        } else {
            // For C, provide a more object-like API using function pointers
            api.push_str(&self.generate_procedural_api()?);
        }

        Ok(api)
    }

    fn generate_hybrid_api(&self) -> Result<String> {
        let mut api = String::new();
        
        // Combine procedural and object-oriented approaches
        api.push_str(&self.generate_procedural_api()?);
        
        if self.config.language == BindingLanguage::Cpp {
            api.push_str(&self.generate_oo_api()?);
        }
        
        Ok(api)
    }

    fn generate_source(&self) -> Result<PathBuf> {
        let source_path = match self.config.language {
            BindingLanguage::C | BindingLanguage::CWithCppWrapper => {
                self.output_dir.join("src").join(format!("{}.c", self.config.library_name))
            }
            BindingLanguage::Cpp => {
                self.output_dir.join("src").join(format!("{}.cpp", self.config.library_name))
            }
        };

        let mut source_content = String::new();

        // Add includes
        source_content.push_str(&format!("#include \"{}.h\"\n", self.config.library_name));
        source_content.push_str("#include <string.h>\n");
        source_content.push_str("#include <stdio.h>\n");
        source_content.push_str("#include <assert.h>\n\n");

        // Add implementation
        source_content.push_str(&self.generate_implementation()?);

        fs::write(&source_path, source_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(source_path)
    }

    fn generate_implementation(&self) -> Result<String> {
        let mut impl_content = String::new();

        // Add static variables and helper functions
        impl_content.push_str(r#"
// === Internal Implementation ===

// Global error context
static scirs2_error_context_t g_error_context = {NULL, NULL};

// Helper function to set error
static void set_error(scirs2_error_t error_code, const char* message) {
    if (g_error_context.callback) {
        g_error_context.callback(error_code, message, g_error_context.user_data);
    }
}

// === Model Functions Implementation ===

scirs2_error_t scirs2_model_load(const char* model_path, scirs2_model_t* model) {
    if (!model_path || !model) {
        set_error(SCIRS2_ERROR_NULL_POINTER, "Null pointer passed to scirs2_model_load");
        return SCIRS2_ERROR_NULL_POINTER;
    }
    
    // Stub implementation - in real code, this would load the actual model
    model->internal_handle = malloc(sizeof(int));
    model->is_valid = (model->internal_handle != NULL);
    
    if (!model->is_valid) {
        set_error(SCIRS2_ERROR_OUT_OF_MEMORY, "Failed to allocate model memory");
        return SCIRS2_ERROR_OUT_OF_MEMORY;
    }
    
    printf("Loading model from: %s\n", model_path);
    return SCIRS2_SUCCESS;
}

scirs2_error_t scirs2_model_load_from_memory(const void* data, size_t size, scirs2_model_t* model) {
    if (!data || !model || size == 0) {
        set_error(SCIRS2_ERROR_NULL_POINTER, "Invalid parameters to scirs2_model_load_from_memory");
        return SCIRS2_ERROR_NULL_POINTER;
    }
    
    // Stub implementation
    model->internal_handle = malloc(sizeof(int));
    model->is_valid = (model->internal_handle != NULL);
    
    if (!model->is_valid) {
        set_error(SCIRS2_ERROR_OUT_OF_MEMORY, "Failed to allocate model memory");
        return SCIRS2_ERROR_OUT_OF_MEMORY;
    }
    
    printf("Loading model from memory (%zu bytes)\n", size);
    return SCIRS2_SUCCESS;
}

void scirs2_model_free(scirs2_model_t* model) {
    if (model && model->internal_handle) {
        free(model->internal_handle);
        model->internal_handle = NULL;
        model->is_valid = false;
    }
}

bool scirs2_model_is_valid(const scirs2_model_t* model) {
    return model && model->is_valid && model->internal_handle;
}

scirs2_error_t scirs2_model_predict(const scirs2_model_t* model, 
                                   const scirs2_tensor_t* input, 
                                   scirs2_tensor_t* output) {
    if (!scirs2_model_is_valid(model)) {
        set_error(SCIRS2_ERROR_INVALID_MODEL, "Invalid model handle");
        return SCIRS2_ERROR_INVALID_MODEL;
    }
    
    if (!input || !output) {
        set_error(SCIRS2_ERROR_NULL_POINTER, "Null tensor pointer");
        return SCIRS2_ERROR_NULL_POINTER;
    }
    
    // Stub implementation - copy input to output
    printf("Running inference\n");
    memcpy(output->data, input->data, input->total_elements * input->element_size);
    
    return SCIRS2_SUCCESS;
}

// === Tensor Functions Implementation ===

scirs2_error_t scirs2_tensor_create(const size_t* shape, size_t ndim, int dtype, scirs2_tensor_t* tensor) {
    if (!shape || !tensor || ndim == 0) {
        set_error(SCIRS2_ERROR_NULL_POINTER, "Invalid parameters to scirs2_tensor_create");
        return SCIRS2_ERROR_NULL_POINTER;
    }
    
    // Calculate total elements
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }
    
    // Allocate tensor structure
    tensor->shape = malloc(ndim * sizeof(size_t));
    if (!tensor->shape) {
        set_error(SCIRS2_ERROR_OUT_OF_MEMORY, "Failed to allocate shape memory");
        return SCIRS2_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    tensor->ndim = ndim;
    tensor->total_elements = total_elements;
    tensor->element_size = sizeof(float); // Assume float for now
    
    // Allocate data
    tensor->data = malloc(total_elements * tensor->element_size);
    if (!tensor->data) {
        free(tensor->shape);
        set_error(SCIRS2_ERROR_OUT_OF_MEMORY, "Failed to allocate tensor data");
        return SCIRS2_ERROR_OUT_OF_MEMORY;
    }
    
    return SCIRS2_SUCCESS;
}

void scirs2_tensor_free(scirs2_tensor_t* tensor) {
    if (tensor) {
        if (tensor->data) {
            free(tensor->data);
            tensor->data = NULL;
        }
        if (tensor->shape) {
            free(tensor->shape);
            tensor->shape = NULL;
        }
        tensor->ndim = 0;
        tensor->total_elements = 0;
        tensor->element_size = 0;
    }
}

void* scirs2_tensor_data(const scirs2_tensor_t* tensor) {
    return tensor ? tensor->data : NULL;
}

// === Utility Functions Implementation ===

const char* scirs2_version(void) {
    return "1.0.0";
}

const char* scirs2_error_message(scirs2_error_t error_code) {
    switch (error_code) {
        case SCIRS2_SUCCESS: return "Success";
        case SCIRS2_ERROR_NULL_POINTER: return "Null pointer error";
        case SCIRS2_ERROR_INVALID_MODEL: return "Invalid model";
        case SCIRS2_ERROR_INVALID_TENSOR: return "Invalid tensor";
        case SCIRS2_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        case SCIRS2_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case SCIRS2_ERROR_IO_ERROR: return "I/O error";
        case SCIRS2_ERROR_COMPUTATION_ERROR: return "Computation error";
        default: return "Unknown error";
    }
}

void scirs2_set_error_callback(scirs2_error_callback_t callback, void* user_data) {
    g_error_context.callback = callback;
    g_error_context.user_data = user_data;
}

"#);

        Ok(impl_content)
    }

    fn generate_cpp_wrapper(&self) -> Result<(PathBuf, PathBuf)> {
        let header_path = self.output_dir.join("include").join(format!("{}_cpp.hpp", self.config.library_name));
        let source_path = self.output_dir.join("src").join(format!("{}_cpp.cpp", self.config.library_name));

        // Generate C++ header
        let header_content = format!(r#"#ifndef {}_CPP_HPP
#define {}_CPP_HPP

#include "{}.h"
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

namespace scirs2 {{

class Exception : public std::runtime_error {{
public:
    explicit Exception(const std::string& message) : std::runtime_error(message) {{}}
    explicit Exception(scirs2_error_t error_code) 
        : std::runtime_error(scirs2_error_message(error_code)) {{}}
}};

class Tensor {{
public:
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const void* data, const std::vector<size_t>& shape, bool copy = true);
    ~Tensor();
    
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    void* data() {{ return tensor_.data; }}
    const void* data() const {{ return tensor_.data; }}
    
    const std::vector<size_t>& shape() const {{ return shape_; }}
    size_t ndim() const {{ return shape_.size(); }}
    size_t size() const;

private:
    scirs2_tensor_t tensor_;
    std::vector<size_t> shape_;
}};

class Model {{
public:
    explicit Model(const std::string& model_path);
    Model(const void* data, size_t size);
    ~Model();
    
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;
    
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    
    bool is_valid() const;
    Tensor predict(const Tensor& input);
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs);

private:
    scirs2_model_t model_;
}};

}} // namespace scirs2

#endif // {}_CPP_HPP
"#, 
            self.config.library_name.to_uppercase(),
            self.config.library_name.to_uppercase(),
            self.config.library_name,
            self.config.library_name.to_uppercase()
        );

        fs::write(&header_path, header_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        // Generate C++ source
        let source_content = format!(r#"#include "{}_cpp.hpp"
#include <cstring>

namespace scirs2 {{

// === Tensor Implementation ===

Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape) {{
    scirs2_error_t result = scirs2_tensor_create(shape.data(), shape.size(), 0, &tensor_);
    if (result != SCIRS2_SUCCESS) {{
        throw Exception(result);
    }}
}}

Tensor::Tensor(const void* data, const std::vector<size_t>& shape, bool copy) 
    : shape_(shape) {{
    scirs2_error_t result = scirs2_tensor_create(shape.data(), shape.size(), 0, &tensor_);
    if (result != SCIRS2_SUCCESS) {{
        throw Exception(result);
    }}
    
    if (copy) {{
        size_t total_size = 1;
        for (size_t dim : shape) total_size *= dim;
        std::memcpy(tensor_.data, data, total_size * sizeof(float));
    }} else {{
        // Take ownership (dangerous - would need more sophisticated memory management)
        tensor_.data = const_cast<void*>(data);
    }}
}}

Tensor::~Tensor() {{
    scirs2_tensor_free(&tensor_);
}}

Tensor::Tensor(Tensor&& other) noexcept : tensor_(other.tensor_), shape_(std::move(other.shape_)) {{
    std::memset(&other.tensor_, 0, sizeof(scirs2_tensor_t));
}}

Tensor& Tensor::operator=(Tensor&& other) noexcept {{
    if (this != &other) {{
        scirs2_tensor_free(&tensor_);
        tensor_ = other.tensor_;
        shape_ = std::move(other.shape_);
        std::memset(&other.tensor_, 0, sizeof(scirs2_tensor_t));
    }}
    return *this;
}}

size_t Tensor::size() const {{
    size_t total = 1;
    for (size_t dim : shape_) total *= dim;
    return total;
}}

// === Model Implementation ===

Model::Model(const std::string& model_path) {{
    scirs2_error_t result = scirs2_model_load(model_path.c_str(), &model_);
    if (result != SCIRS2_SUCCESS) {{
        throw Exception(result);
    }}
}}

Model::Model(const void* data, size_t size) {{
    scirs2_error_t result = scirs2_model_load_from_memory(data, size, &model_);
    if (result != SCIRS2_SUCCESS) {{
        throw Exception(result);
    }}
}}

Model::~Model() {{
    scirs2_model_free(&model_);
}}

Model::Model(Model&& other) noexcept : model_(other.model_) {{
    std::memset(&other.model_, 0, sizeof(scirs2_model_t));
}}

Model& Model::operator=(Model&& other) noexcept {{
    if (this != &other) {{
        scirs2_model_free(&model_);
        model_ = other.model_;
        std::memset(&other.model_, 0, sizeof(scirs2_model_t));
    }}
    return *this;
}}

bool Model::is_valid() const {{
    return scirs2_model_is_valid(&model_);
}}

Tensor Model::predict(const Tensor& input) {{
    Tensor output(input.shape());
    scirs2_error_t result = scirs2_model_predict(&model_, &input.tensor_, &output.tensor_);
    if (result != SCIRS2_SUCCESS) {{
        throw Exception(result);
    }}
    return output;
}}

std::vector<Tensor> Model::predict_batch(const std::vector<Tensor>& inputs) {{
    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());
    
    for (const auto& input : inputs) {{
        outputs.push_back(predict(input));
    }}
    
    return outputs;
}}

}} // namespace scirs2
"#, self.config.library_name);

        fs::write(&source_path, source_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok((header_path, source_path))
    }

    fn generate_build_system(&self) -> Result<Vec<PathBuf>> {
        let mut build_files = Vec::new();

        match self.config.build_system.system {
            BuildSystem::CMake => {
                let cmake_path = self.generate_cmake()?;
                build_files.push(cmake_path);
            }
            BuildSystem::Make => {
                let make_path = self.generate_makefile()?;
                build_files.push(make_path);
            }
            _ => {
                // Generate CMake as default
                let cmake_path = self.generate_cmake()?;
                build_files.push(cmake_path);
            }
        }

        // Generate pkg-config file if requested
        if self.config.build_system.install_config.generate_pkgconfig {
            let pc_path = self.generate_pkgconfig()?;
            build_files.push(pc_path);
        }

        Ok(build_files)
    }

    fn generate_cmake(&self) -> Result<PathBuf> {
        let cmake_path = self.output_dir.join("CMakeLists.txt");
        
        let cmake_content = format!(r#"cmake_minimum_required(VERSION 3.12)
project({} VERSION 1.0.0 LANGUAGES C CXX)

# Set C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# Configure build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/{}.c
)

# C++ wrapper sources (if enabled)
if(ENABLE_CPP_WRAPPER)
    list(APPEND SOURCES src/{}_cpp.cpp)
endif()

# Create shared library
add_library({} SHARED ${{SOURCES}})

# Create static library
add_library({}_static STATIC ${{SOURCES}})
set_target_properties({}_static PROPERTIES OUTPUT_NAME {})

# Set version
set_target_properties({} PROPERTIES 
    VERSION ${{PROJECT_VERSION}}
    SOVERSION 1
)

# Installation
install(TARGETS {} {}_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# Examples
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "{}-config-version.cmake"
    VERSION ${{PACKAGE_VERSION}}
    COMPATIBILITY AnyNewerVersion
)

install(FILES "{}-config.cmake" "${{CMAKE_CURRENT_BINARY_DIR}}/{}-config-version.cmake"
    DESTINATION lib/cmake/{}
)
"#, 
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name
        );

        fs::write(&cmake_path, cmake_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(cmake_path)
    }

    fn generate_makefile(&self) -> Result<PathBuf> {
        let make_path = self.output_dir.join("Makefile");
        
        let make_content = format!(r#"# Makefile for {}

CC = gcc
CXX = g++
CFLAGS = -std=c99 -Wall -Wextra -fPIC -Iinclude
CXXFLAGS = -std=c++17 -Wall -Wextra -fPIC -Iinclude
LDFLAGS = -shared

# Build type
ifdef DEBUG
    CFLAGS += -g -O0 -DDEBUG
    CXXFLAGS += -g -O0 -DDEBUG
else
    CFLAGS += -O3 -DNDEBUG
    CXXFLAGS += -O3 -DNDEBUG
endif

# Directories
SRCDIR = src
INCDIR = include
BUILDDIR = build
LIBDIR = lib

# Sources
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)

# Targets
LIBRARY = $(LIBDIR)/lib{}.so
STATIC_LIB = $(LIBDIR)/lib{}.a

.PHONY: all clean install

all: $(LIBRARY) $(STATIC_LIB)

$(LIBRARY): $(OBJECTS) | $(LIBDIR)
	$(CC) $(LDFLAGS) -o $@ $^

$(STATIC_LIB): $(OBJECTS) | $(LIBDIR)
	ar rcs $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(LIBDIR):
	mkdir -p $(LIBDIR)

clean:
	rm -rf $(BUILDDIR) $(LIBDIR)

install: $(LIBRARY) $(STATIC_LIB)
	install -d $(DESTDIR)/usr/local/lib
	install -d $(DESTDIR)/usr/local/include
	install -m 755 $(LIBRARY) $(DESTDIR)/usr/local/lib/
	install -m 644 $(STATIC_LIB) $(DESTDIR)/usr/local/lib/
	install -m 644 $(INCDIR)/*.h $(DESTDIR)/usr/local/include/

examples: $(LIBRARY)
	$(MAKE) -C examples

tests: $(LIBRARY)
	$(MAKE) -C tests

.SECONDARY: $(OBJECTS)
"#, self.config.library_name, self.config.library_name, self.config.library_name);

        fs::write(&make_path, make_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(make_path)
    }

    fn generate_pkgconfig(&self) -> Result<PathBuf> {
        let pc_path = self.output_dir.join(format!("{}.pc.in", self.config.library_name));
        
        let pc_content = format!(r#"prefix=@CMAKE_INSTALL_PREFIX@
exec_prefix=${{prefix}}
libdir=${{exec_prefix}}/lib
includedir=${{prefix}}/include

Name: {}
Description: SciRS2 Neural Network Library C/C++ Bindings
Version: @PROJECT_VERSION@
Libs: -L${{libdir}} -l{}
Cflags: -I${{includedir}}
"#, self.config.library_name, self.config.library_name);

        fs::write(&pc_path, pc_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(pc_path)
    }

    fn generate_examples(&self) -> Result<Vec<PathBuf>> {
        let mut examples = Vec::new();

        // Generate C example
        let c_example_path = self.output_dir.join("examples").join("basic_usage.c");
        let c_example_content = format!(r#"#include <stdio.h>
#include <stdlib.h>
#include "{}.h"

void error_callback(int error_code, const char* message, void* user_data) {{
    fprintf(stderr, "Error %d: %s\n", error_code, message);
}}

int main() {{
    // Set error callback
    scirs2_set_error_callback(error_callback, NULL);
    
    // Load model
    scirs2_model_t model;
    scirs2_error_t result = scirs2_model_load("model.scirs2", &model);
    if (result != SCIRS2_SUCCESS) {{
        printf("Failed to load model: %s\n", scirs2_error_message(result));
        return 1;
    }}
    
    printf("Model loaded successfully\n");
    printf("Library version: %s\n", scirs2_version());
    
    // Create input tensor
    size_t shape[] = {{1, 10}};
    scirs2_tensor_t input, output;
    
    result = scirs2_tensor_create(shape, 2, 0, &input);
    if (result != SCIRS2_SUCCESS) {{
        printf("Failed to create input tensor: %s\n", scirs2_error_message(result));
        scirs2_model_free(&model);
        return 1;
    }}
    
    result = scirs2_tensor_create(shape, 2, 0, &output);
    if (result != SCIRS2_SUCCESS) {{
        printf("Failed to create output tensor: %s\n", scirs2_error_message(result));
        scirs2_tensor_free(&input);
        scirs2_model_free(&model);
        return 1;
    }}
    
    // Initialize input data
    float* input_data = (float*)scirs2_tensor_data(&input);
    for (int i = 0; i < 10; i++) {{
        input_data[i] = (float)i * 0.1f;
    }}
    
    // Run inference
    result = scirs2_model_predict(&model, &input, &output);
    if (result != SCIRS2_SUCCESS) {{
        printf("Prediction failed: %s\n", scirs2_error_message(result));
    }} else {{
        printf("Prediction successful\n");
        
        // Print output
        float* output_data = (float*)scirs2_tensor_data(&output);
        printf("Output: ");
        for (int i = 0; i < 10; i++) {{
            printf("%.3f ", output_data[i]);
        }}
        printf("\n");
    }}
    
    // Cleanup
    scirs2_tensor_free(&input);
    scirs2_tensor_free(&output);
    scirs2_model_free(&model);
    
    return 0;
}}
"#, self.config.library_name);

        fs::write(&c_example_path, c_example_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;
        examples.push(c_example_path);

        // Generate C++ example if applicable
        if self.config.language == BindingLanguage::Cpp || self.config.language == BindingLanguage::CWithCppWrapper {
            let cpp_example_path = self.output_dir.join("examples").join("basic_usage.cpp");
            let cpp_example_content = format!(r#"#include <iostream>
#include <vector>
#include "{}_cpp.hpp"

int main() {{
    try {{
        // Load model
        scirs2::Model model("model.scirs2");
        
        if (!model.is_valid()) {{
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }}
        
        std::cout << "Model loaded successfully" << std::endl;
        
        // Create input tensor
        std::vector<size_t> shape = {{1, 10}};
        scirs2::Tensor input(shape);
        
        // Initialize input data
        float* input_data = static_cast<float*>(input.data());
        for (size_t i = 0; i < 10; i++) {{
            input_data[i] = static_cast<float>(i) * 0.1f;
        }}
        
        // Run inference
        auto output = model.predict(input);
        
        std::cout << "Prediction successful" << std::endl;
        
        // Print output
        const float* output_data = static_cast<const float*>(output.data());
        std::cout << "Output: ";
        for (size_t i = 0; i < output.size(); i++) {{
            std::cout << output_data[i] << " ";
        }}
        std::cout << std::endl;
        
    }} catch (const scirs2::Exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }}
    
    return 0;
}}
"#, self.config.library_name);

            fs::write(&cpp_example_path, cpp_example_content)
                .map_err(|e| NeuralError::IOError(e.to_string()))?;
            examples.push(cpp_example_path);
        }

        // Generate example CMakeLists.txt
        let example_cmake_path = self.output_dir.join("examples").join("CMakeLists.txt");
        let example_cmake_content = format!(r#"# Examples CMakeLists.txt

# C example
add_executable(basic_usage_c basic_usage.c)
target_link_libraries(basic_usage_c {})

# C++ example (if applicable)
if(ENABLE_CPP_WRAPPER)
    add_executable(basic_usage_cpp basic_usage.cpp)
    target_link_libraries(basic_usage_cpp {})
    set_target_properties(basic_usage_cpp PROPERTIES LINKER_LANGUAGE CXX)
endif()
"#, self.config.library_name, self.config.library_name);

        fs::write(&example_cmake_path, example_cmake_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;
        examples.push(example_cmake_path);

        Ok(examples)
    }

    fn generate_documentation(&self) -> Result<Vec<PathBuf>> {
        let mut docs = Vec::new();

        // Generate README
        let readme_path = self.output_dir.join("README.md");
        let readme_content = format!(r#"# {} - SciRS2 Neural Network C/C++ Bindings

This library provides C/C++ bindings for SciRS2 neural network models.

## Features

- Load and run SciRS2 neural network models
- C and C++ APIs available
- Cross-platform support
- Memory-safe tensor operations
- Error handling with detailed messages

## Building

### Using CMake

```bash
mkdir build
cd build
cmake ..
make
```

### Using Make

```bash
make
```

## Usage

### C API

```c
#include "{}.h"

int main() {{
    scirs2_model_t model;
    scirs2_model_load("model.scirs2", &model);
    
    // Create tensors and run inference
    // ...
    
    scirs2_model_free(&model);
    return 0;
}}
```

### C++ API

```cpp
#include "{}_cpp.hpp"

int main() {{
    try {{
        scirs2::Model model("model.scirs2");
        scirs2::Tensor input({{1, 10}});
        auto output = model.predict(input);
    }} catch (const scirs2::Exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
    }}
    return 0;
}}
```

## API Reference

See the header files in `include/` for detailed API documentation.

## Examples

Check the `examples/` directory for usage examples.

## License

MIT License
"#, self.config.library_name, self.config.library_name, self.config.library_name);

        fs::write(&readme_path, readme_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;
        docs.push(readme_path);

        Ok(docs)
    }
}

impl Default for BindingConfig {
    fn default() -> Self {
        let mut primitives = HashMap::new();
        primitives.insert("f32".to_string(), "float".to_string());
        primitives.insert("f64".to_string(), "double".to_string());
        primitives.insert("i32".to_string(), "int32_t".to_string());
        primitives.insert("u32".to_string(), "uint32_t".to_string());

        Self {
            library_name: "scirs2_model".to_string(),
            language: BindingLanguage::C,
            api_style: ApiStyle::Procedural,
            type_mappings: TypeMappings {
                primitives,
                arrays: ArrayMapping::StructuredArrays,
                strings: StringMapping::CString,
                custom_types: Vec::new(),
            },
            memory_strategy: MemoryStrategy::Manual,
            error_handling: ErrorHandling::ErrorCodes,
            threading: ThreadingConfig {
                safety_level: ThreadSafety::ReadOnly,
                sync_primitives: vec![SyncPrimitive::Mutex],
                thread_pool: None,
            },
            build_system: BuildSystemConfig {
                system: BuildSystem::CMake,
                compiler_flags: vec!["-Wall".to_string(), "-Wextra".to_string()],
                linker_flags: Vec::new(),
                dependencies: Vec::new(),
                install_config: InstallConfig {
                    prefix: "/usr/local".to_string(),
                    bin_dir: "bin".to_string(),
                    lib_dir: "lib".to_string(),
                    include_dir: "include".to_string(),
                    generate_pkgconfig: true,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::sequential::Sequential;
    use crate::layers::dense::Dense;
    use tempfile::TempDir;
    use rand::SeedableRng;

    #[test]
    fn test_binding_config_default() {
        let config = BindingConfig::default();
        assert_eq!(config.library_name, "scirs2_model");
        assert_eq!(config.language, BindingLanguage::C);
        assert_eq!(config.api_style, ApiStyle::Procedural);
    }

    #[test]
    fn test_type_mappings() {
        let config = BindingConfig::default();
        assert!(config.type_mappings.primitives.contains_key("f32"));
        assert_eq!(config.type_mappings.primitives["f32"], "float");
        assert_eq!(config.type_mappings.arrays, ArrayMapping::StructuredArrays);
    }

    #[test]
    fn test_binding_generator_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        
        let mut model = Sequential::new();
        let dense = Dense::new(10, 1, Some("relu"), &mut rng).unwrap();
        model.add_layer(Box::new(dense));

        let config = BindingConfig::default();
        let metadata = PackageMetadata {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            description: "Test".to_string(),
            author: "Test".to_string(),
            license: "MIT".to_string(),
            platforms: vec!["linux".to_string()],
            dependencies: HashMap::new(),
            input_specs: Vec::new(),
            output_specs: Vec::new(),
            runtime_requirements: crate::serving::RuntimeRequirements {
                min_memory_mb: 256,
                cpu_requirements: crate::serving::CpuRequirements {
                    min_cores: 1,
                    instruction_sets: Vec::new(),
                    min_frequency_mhz: None,
                },
                gpu_requirements: None,
                system_dependencies: Vec::new(),
            },
            timestamp: chrono::Utc::now().to_rfc3339(),
            checksum: "test".to_string(),
        };

        let generator = BindingGenerator::new(
            model,
            config,
            metadata,
            temp_dir.path().to_path_buf(),
        );

        assert_eq!(generator.config.library_name, "scirs2_model");
    }

    #[test]
    fn test_custom_type() {
        let custom_type = CustomType {
            rust_name: "MyStruct".to_string(),
            c_name: "my_struct_t".to_string(),
            definition: "typedef struct { int x; float y; } my_struct_t;".to_string(),
            includes: vec!["stdint.h".to_string()],
        };

        assert_eq!(custom_type.rust_name, "MyStruct");
        assert_eq!(custom_type.c_name, "my_struct_t");
        assert!(custom_type.includes.contains(&"stdint.h".to_string()));
    }

    #[test]
    fn test_threading_config() {
        let thread_config = ThreadingConfig {
            safety_level: ThreadSafety::Full,
            sync_primitives: vec![SyncPrimitive::Mutex, SyncPrimitive::RwLock],
            thread_pool: Some(ThreadPoolConfig {
                default_threads: Some(4),
                min_threads: 1,
                max_threads: 16,
                thread_name_pattern: "scirs2-thread-{}".to_string(),
            }),
        };

        assert_eq!(thread_config.safety_level, ThreadSafety::Full);
        assert_eq!(thread_config.sync_primitives.len(), 2);
        assert!(thread_config.thread_pool.is_some());
    }

    #[test]
    fn test_build_system_config() {
        let build_config = BuildSystemConfig {
            system: BuildSystem::CMake,
            compiler_flags: vec!["-O3".to_string(), "-DNDEBUG".to_string()],
            linker_flags: vec!["-lm".to_string()],
            dependencies: vec![Dependency {
                name: "blas".to_string(),
                version: Some("3.0".to_string()),
                headers: vec!["cblas.h".to_string()],
                libraries: vec!["blas".to_string()],
                package_manager: Some("pkg-config".to_string()),
            }],
            install_config: InstallConfig {
                prefix: "/opt/scirs2".to_string(),
                bin_dir: "bin".to_string(),
                lib_dir: "lib".to_string(),
                include_dir: "include".to_string(),
                generate_pkgconfig: true,
            },
        };

        assert_eq!(build_config.system, BuildSystem::CMake);
        assert_eq!(build_config.compiler_flags.len(), 2);
        assert_eq!(build_config.dependencies.len(), 1);
        assert!(build_config.install_config.generate_pkgconfig);
    }
}