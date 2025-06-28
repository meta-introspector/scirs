//! Reverse-mode automatic differentiation (backpropagation)
//!
//! This module implements reverse-mode automatic differentiation for efficient
//! gradient computation in neural networks and optimization algorithms.

use ndarray::{s, Array, Array1, Array2, ArrayBase, Data, Dimension};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};

use crate::error::OptimizerError;

/// Reverse-mode AD engine (gradient tape)
pub struct ReverseModeEngine<T: Float> {
    /// Computation tape for reverse pass
    tape: Vec<ReverseOperation<T>>,

    /// Variable registry
    variables: HashMap<String, usize>,

    /// Gradient storage
    gradients: Vec<Option<Array1<T>>>,

    /// Current recording state
    recording: bool,

    /// Higher-order gradient tracking
    higher_order: bool,

    /// Gradient computation cache
    cache: HashMap<usize, Array1<T>>,
}

/// Reverse-mode operation on the tape
#[derive(Debug, Clone)]
struct ReverseOperation<T: Float> {
    /// Operation type
    op_type: ReverseOpType,

    /// Input variable indices
    inputs: Vec<usize>,

    /// Output variable index
    output: usize,

    /// Backward function for gradient computation
    backward_fn: BackwardFunction<T>,

    /// Saved values for backward pass
    saved_values: SavedValues<T>,
}

/// Reverse operation types
#[derive(Debug, Clone)]
enum ReverseOpType {
    Variable,
    Constant,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    ReLU,
    LeakyReLU,
    MatMul,
    Dot,
    Sum,
    Mean,
    Norm,
    Reshape,
    Transpose,
    Slice,
    Concatenate,
    BatchNorm,
    Dropout,
    Convolution,
}

/// Backward function for computing gradients
#[derive(Debug, Clone)]
enum BackwardFunction<T: Float> {
    /// Identity (for variables and constants)
    Identity,

    /// Addition backward: grad flows through unchanged
    AddBackward,

    /// Subtraction backward: negate gradient for second operand
    SubtractBackward,

    /// Multiplication backward: multiply by other operand
    MultiplyBackward,

    /// Division backward: complex division rule
    DivideBackward,

    /// Power backward: multiply by derivative
    PowerBackward { exponent: T },

    /// Exponential backward
    ExpBackward,

    /// Logarithm backward
    LogBackward,

    /// Trigonometric function backward
    TrigBackward { function: TrigFunction },

    /// Activation function backward
    ActivationBackward { function: ActivationFunction },

    /// Matrix multiplication backward
    MatMulBackward {
        transpose_lhs: bool,
        transpose_rhs: bool,
    },

    /// Reduction backward
    ReductionBackward {
        reduction_type: ReductionType,
        input_shape: Vec<usize>,
        axis: Option<usize>,
    },

    /// Reshape backward
    ReshapeBackward { original_shape: Vec<usize> },

    /// Custom backward function
    Custom { name: String },
}

/// Trigonometric functions
#[derive(Debug, Clone, Copy)]
enum TrigFunction {
    Sin,
    Cos,
    Tan,
}

/// Activation functions
#[derive(Debug, Clone, Copy)]
enum ActivationFunction {
    Tanh,
    Sigmoid,
    ReLU,
    LeakyReLU { alpha: f64 },
}

/// Reduction types
#[derive(Debug, Clone, Copy)]
enum ReductionType {
    Sum,
    Mean,
    Max,
    Min,
    Norm,
}

/// Saved values for backward computation
#[derive(Debug, Clone)]
enum SavedValues<T: Float> {
    None,

    /// Single tensor value
    Tensor(Array1<T>),

    /// Two tensor values
    TensorPair(Array1<T>, Array1<T>),

    /// Scalar value
    Scalar(T),

    /// Shape information
    Shape(Vec<usize>),

    /// Multiple values for complex operations
    Multiple(Vec<Array1<T>>),

    /// Indices for slice operations
    Indices {
        start: Vec<usize>,
        end: Vec<usize>,
    },
}

/// Gradient computation context
#[derive(Debug, Clone)]
pub struct GradientContext<T: Float> {
    /// Variables requiring gradients
    pub requires_grad: HashMap<usize, bool>,

    /// Gradient accumulation mode
    pub accumulate: bool,

    /// Retain computation graph
    pub retain_graph: bool,

    /// Create computation graph
    pub create_graph: bool,
}

impl<T: Float + Default + Clone> Default for GradientContext<T> {
    fn default() -> Self {
        Self {
            requires_grad: HashMap::new(),
            accumulate: true,
            retain_graph: false,
            create_graph: false,
        }
    }
}

impl<T: Float + Default + Clone> ReverseModeEngine<T> {
    /// Create a new reverse-mode AD engine
    pub fn new() -> Self {
        Self {
            tape: Vec::new(),
            variables: HashMap::new(),
            gradients: Vec::new(),
            recording: true,
            higher_order: false,
            cache: HashMap::new(),
        }
    }

    /// Enable/disable gradient recording
    pub fn set_recording(&mut self, recording: bool) {
        self.recording = recording;
    }

    /// Check if recording is enabled
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Enable higher-order gradients
    pub fn enable_higher_order(&mut self, enabled: bool) {
        self.higher_order = enabled;
    }

    /// Clear the computation tape
    pub fn clear_tape(&mut self) {
        self.tape.clear();
        self.gradients.clear();
        self.cache.clear();
    }

    /// Create a variable with gradient tracking
    pub fn create_variable(&mut self, name: &str, value: Array1<T>, requires_grad: bool) -> usize {
        let var_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type: ReverseOpType::Variable,
                inputs: Vec::new(),
                output: var_id,
                backward_fn: BackwardFunction::Identity,
                saved_values: SavedValues::Tensor(value.clone()),
            };

            self.tape.push(op);
        }

        self.variables.insert(name.to_string(), var_id);

        // Initialize gradient storage
        if var_id >= self.gradients.len() {
            self.gradients.resize(var_id + 1, None);
        }

        if requires_grad {
            self.gradients[var_id] = Some(Array1::zeros(value.len()));
        }

        var_id
    }

    /// Create a constant (no gradient tracking)
    pub fn create_constant(&mut self, value: Array1<T>) -> usize {
        let const_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type: ReverseOpType::Constant,
                inputs: Vec::new(),
                output: const_id,
                backward_fn: BackwardFunction::Identity,
                saved_values: SavedValues::Tensor(value),
            };

            self.tape.push(op);
        }

        const_id
    }

    /// Addition operation
    pub fn add(&mut self, lhs: usize, rhs: usize) -> Result<usize, OptimizerError> {
        self.binary_op(ReverseOpType::Add, lhs, rhs, BackwardFunction::AddBackward)
    }

    /// Subtraction operation
    pub fn subtract(&mut self, lhs: usize, rhs: usize) -> Result<usize, OptimizerError> {
        self.binary_op(
            ReverseOpType::Subtract,
            lhs,
            rhs,
            BackwardFunction::SubtractBackward,
        )
    }

    /// Multiplication operation
    pub fn multiply(&mut self, lhs: usize, rhs: usize) -> Result<usize, OptimizerError> {
        // Save operand values for backward pass
        let lhs_val = self.get_value(lhs)?;
        let rhs_val = self.get_value(rhs)?;

        let output_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type: ReverseOpType::Multiply,
                inputs: vec![lhs, rhs],
                output: output_id,
                backward_fn: BackwardFunction::MultiplyBackward,
                saved_values: SavedValues::TensorPair(lhs_val, rhs_val),
            };

            self.tape.push(op);
        }

        Ok(output_id)
    }

    /// Division operation
    pub fn divide(&mut self, lhs: usize, rhs: usize) -> Result<usize, OptimizerError> {
        let lhs_val = self.get_value(lhs)?;
        let rhs_val = self.get_value(rhs)?;

        let output_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type: ReverseOpType::Divide,
                inputs: vec![lhs, rhs],
                output: output_id,
                backward_fn: BackwardFunction::DivideBackward,
                saved_values: SavedValues::TensorPair(lhs_val, rhs_val),
            };

            self.tape.push(op);
        }

        Ok(output_id)
    }

    /// Power operation
    pub fn power(&mut self, base: usize, exponent: T) -> Result<usize, OptimizerError> {
        let base_val = self.get_value(base)?;

        let output_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type: ReverseOpType::Power,
                inputs: vec![base],
                output: output_id,
                backward_fn: BackwardFunction::PowerBackward { exponent },
                saved_values: SavedValues::Tensor(base_val),
            };

            self.tape.push(op);
        }

        Ok(output_id)
    }

    /// Exponential function
    pub fn exp(&mut self, input: usize) -> Result<usize, OptimizerError> {
        self.unary_op(ReverseOpType::Exp, input, BackwardFunction::ExpBackward)
    }

    /// Natural logarithm
    pub fn log(&mut self, input: usize) -> Result<usize, OptimizerError> {
        self.unary_op(ReverseOpType::Log, input, BackwardFunction::LogBackward)
    }

    /// Sine function
    pub fn sin(&mut self, input: usize) -> Result<usize, OptimizerError> {
        self.unary_op(
            ReverseOpType::Sin,
            input,
            BackwardFunction::TrigBackward {
                function: TrigFunction::Sin,
            },
        )
    }

    /// Cosine function
    pub fn cos(&mut self, input: usize) -> Result<usize, OptimizerError> {
        self.unary_op(
            ReverseOpType::Cos,
            input,
            BackwardFunction::TrigBackward {
                function: TrigFunction::Cos,
            },
        )
    }

    /// Hyperbolic tangent
    pub fn tanh(&mut self, input: usize) -> Result<usize, OptimizerError> {
        self.unary_op(
            ReverseOpType::Tanh,
            input,
            BackwardFunction::ActivationBackward {
                function: ActivationFunction::Tanh,
            },
        )
    }

    /// Sigmoid function
    pub fn sigmoid(&mut self, input: usize) -> Result<usize, OptimizerError> {
        self.unary_op(
            ReverseOpType::Sigmoid,
            input,
            BackwardFunction::ActivationBackward {
                function: ActivationFunction::Sigmoid,
            },
        )
    }

    /// ReLU function
    pub fn relu(&mut self, input: usize) -> Result<usize, OptimizerError> {
        self.unary_op(
            ReverseOpType::ReLU,
            input,
            BackwardFunction::ActivationBackward {
                function: ActivationFunction::ReLU,
            },
        )
    }

    /// Leaky ReLU function
    pub fn leaky_relu(&mut self, input: usize, alpha: f64) -> Result<usize, OptimizerError> {
        self.unary_op(
            ReverseOpType::LeakyReLU,
            input,
            BackwardFunction::ActivationBackward {
                function: ActivationFunction::LeakyReLU { alpha },
            },
        )
    }

    /// Matrix multiplication
    pub fn matmul(&mut self, lhs: usize, rhs: usize) -> Result<usize, OptimizerError> {
        let lhs_val = self.get_value(lhs)?;
        let rhs_val = self.get_value(rhs)?;

        let output_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type: ReverseOpType::MatMul,
                inputs: vec![lhs, rhs],
                output: output_id,
                backward_fn: BackwardFunction::MatMulBackward {
                    transpose_lhs: false,
                    transpose_rhs: false,
                },
                saved_values: SavedValues::TensorPair(lhs_val, rhs_val),
            };

            self.tape.push(op);
        }

        Ok(output_id)
    }

    /// Dot product
    pub fn dot(&mut self, lhs: usize, rhs: usize) -> Result<usize, OptimizerError> {
        self.binary_op(
            ReverseOpType::Dot,
            lhs,
            rhs,
            BackwardFunction::MultiplyBackward,
        )
    }

    /// Sum reduction
    pub fn sum(&mut self, input: usize, axis: Option<usize>) -> Result<usize, OptimizerError> {
        let input_val = self.get_value(input)?;
        let input_shape = input_val.shape().to_vec();

        let output_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type: ReverseOpType::Sum,
                inputs: vec![input],
                output: output_id,
                backward_fn: BackwardFunction::ReductionBackward {
                    reduction_type: ReductionType::Sum,
                    input_shape,
                    axis,
                },
                saved_values: SavedValues::None,
            };

            self.tape.push(op);
        }

        Ok(output_id)
    }

    /// Mean reduction
    pub fn mean(&mut self, input: usize, axis: Option<usize>) -> Result<usize, OptimizerError> {
        let input_val = self.get_value(input)?;
        let input_shape = input_val.shape().to_vec();

        let output_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type: ReverseOpType::Mean,
                inputs: vec![input],
                output: output_id,
                backward_fn: BackwardFunction::ReductionBackward {
                    reduction_type: ReductionType::Mean,
                    input_shape,
                    axis,
                },
                saved_values: SavedValues::None,
            };

            self.tape.push(op);
        }

        Ok(output_id)
    }

    /// L2 norm
    pub fn norm(&mut self, input: usize) -> Result<usize, OptimizerError> {
        let input_val = self.get_value(input)?;
        let input_shape = input_val.shape().to_vec();

        let output_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type: ReverseOpType::Norm,
                inputs: vec![input],
                output: output_id,
                backward_fn: BackwardFunction::ReductionBackward {
                    reduction_type: ReductionType::Norm,
                    input_shape,
                    axis: None,
                },
                saved_values: SavedValues::Tensor(input_val),
            };

            self.tape.push(op);
        }

        Ok(output_id)
    }

    /// Reshape operation
    pub fn reshape(&mut self, input: usize, new_shape: &[usize]) -> Result<usize, OptimizerError> {
        let input_val = self.get_value(input)?;
        let original_shape = input_val.shape().to_vec();

        let output_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type: ReverseOpType::Reshape,
                inputs: vec![input],
                output: output_id,
                backward_fn: BackwardFunction::ReshapeBackward { original_shape },
                saved_values: SavedValues::Shape(new_shape.to_vec()),
            };

            self.tape.push(op);
        }

        Ok(output_id)
    }

    /// Backward pass - compute gradients
    pub fn backward(
        &mut self,
        output_id: usize,
        gradient: Option<Array1<T>>,
    ) -> Result<(), OptimizerError> {
        // Initialize output gradient
        if output_id >= self.gradients.len() {
            self.gradients.resize(output_id + 1, None);
        }

        let output_grad = gradient.unwrap_or_else(|| Array1::ones(1));
        self.gradients[output_id] = Some(output_grad);

        // Reverse pass through the tape
        for op in self.tape.iter().rev() {
            if let Some(ref output_gradient) = self.gradients[op.output] {
                let input_gradients = self.compute_backward_pass(op, output_gradient)?;

                // Accumulate gradients for inputs
                for (i, &input_id) in op.inputs.iter().enumerate() {
                    if i < input_gradients.len() {
                        if input_id >= self.gradients.len() {
                            self.gradients.resize(input_id + 1, None);
                        }

                        if let Some(ref mut existing_grad) = self.gradients[input_id] {
                            *existing_grad = &*existing_grad + &input_gradients[i];
                        } else {
                            self.gradients[input_id] = Some(input_gradients[i].clone());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get gradient for a variable
    pub fn get_gradient(&self, var_id: usize) -> Option<&Array1<T>> {
        self.gradients.get(var_id)?.as_ref()
    }

    /// Get gradient by variable name
    pub fn get_gradient_by_name(&self, name: &str) -> Option<&Array1<T>> {
        let var_id = *self.variables.get(name)?;
        self.get_gradient(var_id)
    }

    /// Zero all gradients
    pub fn zero_gradients(&mut self) {
        for gradient in &mut self.gradients {
            if let Some(ref mut grad) = gradient {
                grad.fill(T::zero());
            }
        }
        self.cache.clear();
    }

    /// Get all variable gradients
    pub fn get_all_gradients(&self) -> HashMap<String, Array1<T>> {
        let mut result = HashMap::new();

        for (name, &var_id) in &self.variables {
            if let Some(grad) = self.get_gradient(var_id) {
                result.insert(name.clone(), grad.clone());
            }
        }

        result
    }

    fn binary_op(
        &mut self,
        op_type: ReverseOpType,
        lhs: usize,
        rhs: usize,
        backward_fn: BackwardFunction<T>,
    ) -> Result<usize, OptimizerError> {
        let output_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type,
                inputs: vec![lhs, rhs],
                output: output_id,
                backward_fn,
                saved_values: SavedValues::None,
            };

            self.tape.push(op);
        }

        Ok(output_id)
    }

    fn unary_op(
        &mut self,
        op_type: ReverseOpType,
        input: usize,
        backward_fn: BackwardFunction<T>,
    ) -> Result<usize, OptimizerError> {
        let input_val = self.get_value(input)?;

        let output_id = self.tape.len();

        if self.recording {
            let op = ReverseOperation {
                op_type,
                inputs: vec![input],
                output: output_id,
                backward_fn,
                saved_values: SavedValues::Tensor(input_val),
            };

            self.tape.push(op);
        }

        Ok(output_id)
    }

    fn get_value(&self, var_id: usize) -> Result<Array1<T>, OptimizerError> {
        if var_id >= self.tape.len() {
            return Err(OptimizerError::InvalidConfig(
                "Invalid variable ID".to_string(),
            ));
        }

        match &self.tape[var_id].saved_values {
            SavedValues::Tensor(tensor) => Ok(tensor.clone()),
            _ => {
                // For operations, we'd need to compute the forward value
                // This is simplified - in practice would need full forward evaluation
                Ok(Array1::zeros(1))
            }
        }
    }

    fn compute_backward_pass(
        &self,
        op: &ReverseOperation<T>,
        output_grad: &Array1<T>,
    ) -> Result<Vec<Array1<T>>, OptimizerError> {
        match &op.backward_fn {
            BackwardFunction::Identity => Ok(vec![output_grad.clone()]),

            BackwardFunction::AddBackward => {
                // Addition: gradients flow through unchanged
                Ok(vec![output_grad.clone(), output_grad.clone()])
            }

            BackwardFunction::SubtractBackward => {
                // Subtraction: negate gradient for second operand
                Ok(vec![output_grad.clone(), -output_grad])
            }

            BackwardFunction::MultiplyBackward => {
                if let SavedValues::TensorPair(ref lhs_val, ref rhs_val) = op.saved_values {
                    // Multiplication: multiply gradient by other operand
                    let lhs_grad = output_grad * rhs_val;
                    let rhs_grad = output_grad * lhs_val;
                    Ok(vec![lhs_grad, rhs_grad])
                } else {
                    Err(OptimizerError::InvalidConfig(
                        "Missing saved values for multiplication".to_string(),
                    ))
                }
            }

            BackwardFunction::DivideBackward => {
                if let SavedValues::TensorPair(ref lhs_val, ref rhs_val) = op.saved_values {
                    // Division: (u/v)' = (u'*v - u*v') / v^2
                    let lhs_grad = output_grad / rhs_val;
                    let rhs_grad = -output_grad * lhs_val / (rhs_val * rhs_val);
                    Ok(vec![lhs_grad, rhs_grad])
                } else {
                    Err(OptimizerError::InvalidConfig(
                        "Missing saved values for division".to_string(),
                    ))
                }
            }

            BackwardFunction::PowerBackward { exponent } => {
                if let SavedValues::Tensor(ref base_val) = op.saved_values {
                    // Power: (u^n)' = n * u^(n-1) * u'
                    let derivative = base_val.mapv(|x| *exponent * x.powf(*exponent - T::one()));
                    let grad = output_grad * &derivative;
                    Ok(vec![grad])
                } else {
                    Err(OptimizerError::InvalidConfig(
                        "Missing saved values for power".to_string(),
                    ))
                }
            }

            BackwardFunction::ExpBackward => {
                if let SavedValues::Tensor(ref input_val) = op.saved_values {
                    // Exponential: (e^u)' = e^u * u'
                    let exp_val = input_val.mapv(|x| x.exp());
                    let grad = output_grad * &exp_val;
                    Ok(vec![grad])
                } else {
                    Err(OptimizerError::InvalidConfig(
                        "Missing saved values for exp".to_string(),
                    ))
                }
            }

            BackwardFunction::LogBackward => {
                if let SavedValues::Tensor(ref input_val) = op.saved_values {
                    // Logarithm: (ln(u))' = u' / u
                    let grad = output_grad / input_val;
                    Ok(vec![grad])
                } else {
                    Err(OptimizerError::InvalidConfig(
                        "Missing saved values for log".to_string(),
                    ))
                }
            }

            BackwardFunction::TrigBackward { function } => {
                if let SavedValues::Tensor(ref input_val) = op.saved_values {
                    let derivative = match function {
                        TrigFunction::Sin => input_val.mapv(|x| x.cos()), // cos(u)
                        TrigFunction::Cos => input_val.mapv(|x| -x.sin()), // -sin(u)
                        TrigFunction::Tan => input_val.mapv(|x| T::one() + x.tan().powi(2)), // sec^2(u)
                    };
                    let grad = output_grad * &derivative;
                    Ok(vec![grad])
                } else {
                    Err(OptimizerError::InvalidConfig(
                        "Missing saved values for trig function".to_string(),
                    ))
                }
            }

            BackwardFunction::ActivationBackward { function } => {
                if let SavedValues::Tensor(ref input_val) = op.saved_values {
                    let derivative = match function {
                        ActivationFunction::Tanh => {
                            let tanh_val = input_val.mapv(|x| x.tanh());
                            tanh_val.mapv(|y| T::one() - y * y)
                        }
                        ActivationFunction::Sigmoid => {
                            let sigmoid_val =
                                input_val.mapv(|x| T::one() / (T::one() + (-x).exp()));
                            sigmoid_val.mapv(|y| y * (T::one() - y))
                        }
                        ActivationFunction::ReLU => {
                            input_val.mapv(|x| if x > T::zero() { T::one() } else { T::zero() })
                        }
                        ActivationFunction::LeakyReLU { alpha } => {
                            let alpha_t = T::from(*alpha).unwrap();
                            input_val.mapv(|x| if x > T::zero() { T::one() } else { alpha_t })
                        }
                    };
                    let grad = output_grad * &derivative;
                    Ok(vec![grad])
                } else {
                    Err(OptimizerError::InvalidConfig(
                        "Missing saved values for activation".to_string(),
                    ))
                }
            }

            BackwardFunction::ReductionBackward {
                reduction_type,
                input_shape,
                axis: _,
            } => {
                match reduction_type {
                    ReductionType::Sum => {
                        // Sum: gradient broadcasts to input shape
                        let grad = Array1::from_elem(input_shape[0], output_grad[0]);
                        Ok(vec![grad])
                    }
                    ReductionType::Mean => {
                        // Mean: gradient divided by input size then broadcast
                        let n = T::from(input_shape[0]).unwrap();
                        let grad = Array1::from_elem(input_shape[0], output_grad[0] / n);
                        Ok(vec![grad])
                    }
                    ReductionType::Norm => {
                        if let SavedValues::Tensor(ref input_val) = op.saved_values {
                            // Norm: gradient is input / norm
                            let norm = input_val.iter().map(|&x| x * x).sum::<T>().sqrt();
                            let grad = if norm > T::zero() {
                                input_val * (output_grad[0] / norm)
                            } else {
                                Array1::zeros(input_val.len())
                            };
                            Ok(vec![grad])
                        } else {
                            Err(OptimizerError::InvalidConfig(
                                "Missing saved values for norm".to_string(),
                            ))
                        }
                    }
                    _ => Ok(vec![output_grad.clone()]),
                }
            }

            BackwardFunction::ReshapeBackward { original_shape } => {
                // Reshape: just reshape gradient back to original shape
                let total_elements: usize = original_shape.iter().product();
                if output_grad.len() == total_elements {
                    Ok(vec![output_grad.clone()])
                } else {
                    // Create appropriately sized gradient
                    let grad = Array1::zeros(original_shape[0]);
                    Ok(vec![grad])
                }
            }

            _ => {
                // Default: pass gradient through unchanged
                Ok(vec![output_grad.clone()])
            }
        }
    }

    /// Get tape statistics
    pub fn get_tape_stats(&self) -> ReverseModeStats {
        ReverseModeStats {
            tape_length: self.tape.len(),
            num_variables: self.variables.len(),
            num_gradients: self.gradients.iter().filter(|g| g.is_some()).count(),
            memory_usage_estimate: self.estimate_memory_usage(),
            cache_size: self.cache.len(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        let tape_size = self.tape.len() * std::mem::size_of::<ReverseOperation<T>>();
        let gradient_size = self
            .gradients
            .iter()
            .filter_map(|g| g.as_ref())
            .map(|g| g.len() * std::mem::size_of::<T>())
            .sum::<usize>();

        tape_size + gradient_size
    }
}

/// Reverse-mode AD statistics
#[derive(Debug, Clone)]
pub struct ReverseModeStats {
    pub tape_length: usize,
    pub num_variables: usize,
    pub num_gradients: usize,
    pub memory_usage_estimate: usize,
    pub cache_size: usize,
}

/// Gradient accumulation utilities
pub struct GradientAccumulator<T: Float> {
    /// Accumulated gradients
    gradients: HashMap<String, Array1<T>>,

    /// Accumulation count
    count: usize,
}

impl<T: Float + Default + Clone> GradientAccumulator<T> {
    pub fn new() -> Self {
        Self {
            gradients: HashMap::new(),
            count: 0,
        }
    }

    /// Accumulate gradients
    pub fn accumulate(&mut self, gradients: HashMap<String, Array1<T>>) {
        for (name, grad) in gradients {
            if let Some(existing) = self.gradients.get_mut(&name) {
                *existing = &*existing + &grad;
            } else {
                self.gradients.insert(name, grad);
            }
        }
        self.count += 1;
    }

    /// Get averaged gradients
    pub fn get_averaged_gradients(&self) -> HashMap<String, Array1<T>> {
        if self.count == 0 {
            return HashMap::new();
        }

        let count_t = T::from(self.count).unwrap();
        self.gradients
            .iter()
            .map(|(name, grad)| (name.clone(), grad / count_t))
            .collect()
    }

    /// Clear accumulated gradients
    pub fn clear(&mut self) {
        self.gradients.clear();
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_mode_engine_creation() {
        let engine = ReverseModeEngine::<f64>::new();
        assert!(engine.is_recording());
        assert_eq!(engine.tape.len(), 0);
    }

    #[test]
    fn test_variable_creation() {
        let mut engine = ReverseModeEngine::new();
        let x_val = Array1::from_vec(vec![2.0, 3.0]);
        let x_id = engine.create_variable("x", x_val, true);

        assert_eq!(x_id, 0);
        assert!(engine.variables.contains_key("x"));
        assert!(engine.get_gradient(x_id).is_some());
    }

    #[test]
    fn test_simple_backward_pass() {
        let mut engine = ReverseModeEngine::new();

        let x_val = Array1::from_vec(vec![2.0]);
        let x_id = engine.create_variable("x", x_val, true);

        let y_val = Array1::from_vec(vec![3.0]);
        let y_id = engine.create_variable("y", y_val, true);

        let sum_id = engine.add(x_id, y_id).unwrap();

        let output_grad = Array1::from_vec(vec![1.0]);
        engine.backward(sum_id, Some(output_grad)).unwrap();

        let x_grad = engine.get_gradient(x_id).unwrap();
        let y_grad = engine.get_gradient(y_id).unwrap();

        assert_eq!(x_grad[0], 1.0);
        assert_eq!(y_grad[0], 1.0);
    }

    #[test]
    fn test_gradient_accumulator() {
        let mut accumulator = GradientAccumulator::<f64>::new();

        let mut grad1 = HashMap::new();
        grad1.insert("x".to_string(), Array1::from_vec(vec![1.0, 2.0]));

        let mut grad2 = HashMap::new();
        grad2.insert("x".to_string(), Array1::from_vec(vec![3.0, 4.0]));

        accumulator.accumulate(grad1);
        accumulator.accumulate(grad2);

        let averaged = accumulator.get_averaged_gradients();
        let x_avg = &averaged["x"];

        assert_eq!(x_avg[0], 2.0); // (1 + 3) / 2
        assert_eq!(x_avg[1], 3.0); // (2 + 4) / 2
    }

    #[test]
    fn test_tape_statistics() {
        let mut engine = ReverseModeEngine::new();

        let x_val = Array1::from_vec(vec![1.0]);
        let x_id = engine.create_variable("x", x_val, true);

        let _exp_id = engine.exp(x_id).unwrap();

        let stats = engine.get_tape_stats();
        assert_eq!(stats.tape_length, 2);
        assert_eq!(stats.num_variables, 1);
    }
}
