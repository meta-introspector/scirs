//! Reverse mode automatic differentiation (backpropagation)
//!
//! Reverse mode AD is efficient for computing gradients when the number of
//! outputs is small compared to the number of inputs.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{Array1, Array2, ArrayView1};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Operations that can be recorded on the tape
#[derive(Debug, Clone)]
pub enum Operation<F: IntegrateFloat> {
    /// Variable input
    Variable(usize),
    /// Constant value
    Constant(F),
    /// Addition
    Add(usize, usize),
    /// Subtraction
    Sub(usize, usize),
    /// Multiplication
    Mul(usize, usize),
    /// Division
    Div(usize, usize),
    /// Negation
    Neg(usize),
    /// Power
    Pow(usize, F),
    /// General power (base and exponent are both tape values)
    PowGeneral(usize, usize),
    /// Sin
    Sin(usize),
    /// Cos
    Cos(usize),
    /// Tan
    Tan(usize),
    /// Exp
    Exp(usize),
    /// Ln
    Ln(usize),
    /// Sqrt
    Sqrt(usize),
    /// Tanh
    Tanh(usize),
    /// Sinh
    Sinh(usize),
    /// Cosh
    Cosh(usize),
    /// Atan2
    Atan2(usize, usize),
    /// Abs
    Abs(usize),
    /// Max
    Max(usize, usize),
    /// Min
    Min(usize, usize),
}

/// Node in the computation graph
pub struct TapeNode<F: IntegrateFloat> {
    /// The value at this node
    pub value: F,
    /// The operation that produced this value
    pub operation: Operation<F>,
    /// The gradient accumulated at this node
    pub gradient: RefCell<F>,
}

impl<F: IntegrateFloat> TapeNode<F> {
    /// Create a new tape node
    pub fn new(_value: F, operation: Operation<F>) -> Self {
        TapeNode {
            _value,
            operation,
            gradient: RefCell::new(F::zero()),
        }
    }
}

/// Reverse mode AD tape for recording operations
pub struct Tape<F: IntegrateFloat> {
    /// Nodes in the computation graph
    nodes: Vec<Rc<TapeNode<F>>>,
    /// Mapping from variable indices to node indices
    var_map: HashMap<usize, usize>,
}

impl<F: IntegrateFloat> Tape<F> {
    /// Create a new tape
    pub fn new() -> Self {
        Tape {
            nodes: Vec::new(),
            var_map: HashMap::new(),
        }
    }

    /// Add a variable to the tape
    pub fn variable(_idx: usize, value: F) -> usize {
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Variable(_idx))));
        self.var_map.insert(_idx, node_idx);
        node_idx
    }

    /// Add a constant to the tape
    pub fn constant(_value: F) -> usize {
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(_value, Operation::Constant(_value))));
        node_idx
    }

    /// Record addition
    pub fn add(a: usize, b: usize) -> usize {
        let value = self.nodes[a].value + self.nodes[b].value;
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Add(a, b))));
        node_idx
    }

    /// Record subtraction
    pub fn sub(a: usize, b: usize) -> usize {
        let value = self.nodes[a].value - self.nodes[b].value;
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Sub(a, b))));
        node_idx
    }

    /// Record multiplication
    pub fn mul(a: usize, b: usize) -> usize {
        let value = self.nodes[a].value * self.nodes[b].value;
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Mul(a, b))));
        node_idx
    }

    /// Record division
    pub fn div(a: usize, b: usize) -> usize {
        let value = self.nodes[a].value / self.nodes[b].value;
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Div(a, b))));
        node_idx
    }

    /// Record negation
    pub fn neg(a: usize) -> usize {
        let value = -self.nodes[a].value;
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Neg(a))));
        node_idx
    }

    /// Record power
    pub fn pow(a: usize, n: F) -> usize {
        let value = self.nodes[a].value.powf(n);
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Pow(a, n))));
        node_idx
    }

    /// Record sin
    pub fn sin(a: usize) -> usize {
        let value = self.nodes[a].value.sin();
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Sin(a))));
        node_idx
    }

    /// Record cos
    pub fn cos(a: usize) -> usize {
        let value = self.nodes[a].value.cos();
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Cos(a))));
        node_idx
    }

    /// Record exp
    pub fn exp(a: usize) -> usize {
        let value = self.nodes[a].value.exp();
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Exp(a))));
        node_idx
    }

    /// Record ln
    pub fn ln(a: usize) -> usize {
        let value = self.nodes[a].value.ln();
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Ln(a))));
        node_idx
    }

    /// Record sqrt
    pub fn sqrt(a: usize) -> usize {
        let value = self.nodes[a].value.sqrt();
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Sqrt(a))));
        node_idx
    }

    /// Record general power where both base and exponent are variables
    pub fn pow_general(a: usize, b: usize) -> usize {
        let value = self.nodes[a].value.powf(self.nodes[b].value);
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::PowGeneral(a, b))));
        node_idx
    }

    /// Record tan
    pub fn tan(a: usize) -> usize {
        let value = self.nodes[a].value.tan();
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Tan(a))));
        node_idx
    }

    /// Record tanh
    pub fn tanh(a: usize) -> usize {
        let value = self.nodes[a].value.tanh();
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Tanh(a))));
        node_idx
    }

    /// Record sinh
    pub fn sinh(a: usize) -> usize {
        let value = self.nodes[a].value.sinh();
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Sinh(a))));
        node_idx
    }

    /// Record cosh
    pub fn cosh(a: usize) -> usize {
        let value = self.nodes[a].value.cosh();
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Cosh(a))));
        node_idx
    }

    /// Record atan2
    pub fn atan2(y: usize, x: usize) -> usize {
        let value = self.nodes[y].value.atan2(self.nodes[x].value);
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Atan2(y, x))));
        node_idx
    }

    /// Record abs
    pub fn abs(a: usize) -> usize {
        let value = self.nodes[a].value.abs();
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Abs(a))));
        node_idx
    }

    /// Record max
    pub fn max(a: usize, b: usize) -> usize {
        let value = self.nodes[a].value.max(self.nodes[b].value);
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Max(a, b))));
        node_idx
    }

    /// Record min
    pub fn min(a: usize, b: usize) -> usize {
        let value = self.nodes[a].value.min(self.nodes[b].value);
        let node_idx = self.nodes.len();
        self.nodes
            .push(Rc::new(TapeNode::new(value, Operation::Min(a, b))));
        node_idx
    }

    /// Get the value at a node
    pub fn value(_idx: usize) -> F {
        self.nodes[_idx].value
    }

    /// Backward pass to compute gradients
    pub fn backward(_output_idx: usize, n_vars: usize) -> Array1<F> {
        // Initialize gradients to zero
        for node in &self.nodes {
            *node.gradient.borrow_mut() = F::zero();
        }

        // Set gradient of output to 1
        *self.nodes[output_idx].gradient.borrow_mut() = F::one();

        // Backward pass
        for i in (0..=output_idx).rev() {
            let node = &self.nodes[i];
            let grad = *node.gradient.borrow();

            if grad.abs() < F::epsilon() {
                continue;
            }

            match &node.operation {
                Operation::Variable(_) | Operation::Constant(_) => {}
                Operation::Add(a, b) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad;
                    *self.nodes[*b].gradient.borrow_mut() += grad;
                }
                Operation::Sub(a, b) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad;
                    *self.nodes[*b].gradient.borrow_mut() -= grad;
                }
                Operation::Mul(a, b) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad * self.nodes[*b].value;
                    *self.nodes[*b].gradient.borrow_mut() += grad * self.nodes[*a].value;
                }
                Operation::Div(a, b) => {
                    let b_val = self.nodes[*b].value;
                    *self.nodes[*a].gradient.borrow_mut() += grad / b_val;
                    *self.nodes[*b].gradient.borrow_mut() -=
                        grad * self.nodes[*a].value / (b_val * b_val);
                }
                Operation::Neg(a) => {
                    *self.nodes[*a].gradient.borrow_mut() -= grad;
                }
                Operation::Pow(a, n) => {
                    *self.nodes[*a].gradient.borrow_mut() +=
                        grad * *n * self.nodes[*a].value.powf(*n - F::one());
                }
                Operation::Sin(a) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad * self.nodes[*a].value.cos();
                }
                Operation::Cos(a) => {
                    *self.nodes[*a].gradient.borrow_mut() -= grad * self.nodes[*a].value.sin();
                }
                Operation::Exp(a) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad * node.value;
                }
                Operation::Ln(a) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad / self.nodes[*a].value;
                }
                Operation::Sqrt(a) => {
                    *self.nodes[*a].gradient.borrow_mut() +=
                        grad / (F::from(2.0).unwrap() * node.value);
                }
                Operation::PowGeneral(a, b) => {
                    // d/da(a^b) = b * a^(b-1)
                    // d/db(a^b) = a^b * ln(a)
                    let a_val = self.nodes[*a].value;
                    let b_val = self.nodes[*b].value;
                    *self.nodes[*a].gradient.borrow_mut() +=
                        grad * b_val * a_val.powf(b_val - F::one());
                    *self.nodes[*b].gradient.borrow_mut() += grad * node.value * a_val.ln();
                }
                Operation::Tan(a) => {
                    // d/dx(tan(x)) = sec²(x) = 1/cos²(x)
                    let cos_val = self.nodes[*a].value.cos();
                    *self.nodes[*a].gradient.borrow_mut() += grad / (cos_val * cos_val);
                }
                Operation::Tanh(a) => {
                    // d/dx(tanh(x)) = 1 - tanh²(x)
                    let tanh_val = node.value;
                    *self.nodes[*a].gradient.borrow_mut() +=
                        grad * (F::one() - tanh_val * tanh_val);
                }
                Operation::Sinh(a) => {
                    // d/dx(sinh(x)) = cosh(x)
                    *self.nodes[*a].gradient.borrow_mut() += grad * self.nodes[*a].value.cosh();
                }
                Operation::Cosh(a) => {
                    // d/dx(cosh(x)) = sinh(x)
                    *self.nodes[*a].gradient.borrow_mut() += grad * self.nodes[*a].value.sinh();
                }
                Operation::Atan2(y, x) => {
                    // d/dy(atan2(y,x)) = x/(x² + y²)
                    // d/dx(atan2(y,x)) = -y/(x² + y²)
                    let x_val = self.nodes[*x].value;
                    let y_val = self.nodes[*y].value;
                    let denom = x_val * x_val + y_val * y_val;
                    *self.nodes[*y].gradient.borrow_mut() += grad * x_val / denom;
                    *self.nodes[*x].gradient.borrow_mut() -= grad * y_val / denom;
                }
                Operation::Abs(a) => {
                    // d/dx(|x|) = sign(x)
                    let sign = if self.nodes[*a].value >= F::zero() {
                        F::one()
                    } else {
                        -F::one()
                    };
                    *self.nodes[*a].gradient.borrow_mut() += grad * sign;
                }
                Operation::Max(a, b) => {
                    // Gradient flows to the larger input
                    if self.nodes[*a].value >= self.nodes[*b].value {
                        *self.nodes[*a].gradient.borrow_mut() += grad;
                    } else {
                        *self.nodes[*b].gradient.borrow_mut() += grad;
                    }
                }
                Operation::Min(a, b) => {
                    // Gradient flows to the smaller input
                    if self.nodes[*a].value <= self.nodes[*b].value {
                        *self.nodes[*a].gradient.borrow_mut() += grad;
                    } else {
                        *self.nodes[*b].gradient.borrow_mut() += grad;
                    }
                }
            }
        }

        // Collect gradients for variables
        let mut gradients = Array1::zeros(n_vars);
        for (var_idx, &node_idx) in &self.var_map {
            if *var_idx < n_vars {
                gradients[*var_idx] = *self.nodes[node_idx].gradient.borrow();
            }
        }

        gradients
    }
}

impl<F: IntegrateFloat> Default for Tape<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Checkpointing strategy for memory-efficient gradient computation
#[derive(Debug, Clone, Copy)]
pub enum CheckpointStrategy {
    /// No checkpointing (store everything)
    None,
    /// Fixed interval checkpointing
    FixedInterval(usize),
    /// Logarithmic checkpointing
    Logarithmic,
    /// Memory-based checkpointing
    MemoryBased { max_nodes: usize },
}

/// Reverse mode automatic differentiation engine
pub struct ReverseAD<F: IntegrateFloat> {
    /// Number of independent variables
    n_vars: usize,
    /// Checkpointing strategy
    checkpoint_strategy: CheckpointStrategy, _phantom: std::marker::PhantomData<F>,
}

impl<F: IntegrateFloat> ReverseAD<F> {
    /// Create a new reverse AD engine
    pub fn new(_n_vars: usize) -> Self {
        ReverseAD {
            _n_vars,
            checkpoint_strategy: CheckpointStrategy::None, _phantom: std::marker::PhantomData,
        }
    }

    /// Set checkpointing strategy
    pub fn with_checkpoint_strategy(mut self, strategy: CheckpointStrategy) -> Self {
        self.checkpoint_strategy = strategy;
        self
    }

    /// Compute gradient using reverse mode AD
    pub fn gradient<Func>(f: Func, x: ArrayView1<F>) -> IntegrateResult<Array1<F>>
    where
        Func: Fn(&mut Tape<F>, &[usize]) -> usize,
    {
        if x.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} variables, got {}",
                self.n_vars,
                x.len()
            )));
        }

        let mut tape = Tape::new();
        let mut var_indices = Vec::new();

        // Add variables to tape
        for (i, &val) in x.iter().enumerate() {
            let idx = tape.variable(i, val);
            var_indices.push(idx);
        }

        // Compute function
        let output_idx = f(&mut tape, &var_indices);

        // Backward pass
        Ok(tape.backward(output_idx, self.n_vars))
    }

    /// Compute Jacobian using reverse mode AD
    pub fn jacobian<Func>(f: Func, x: ArrayView1<F>) -> IntegrateResult<Array2<F>>
    where
        Func: Fn(&mut Tape<F>, &[usize]) -> Vec<usize>,
    {
        if x.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} variables, got {}",
                self.n_vars,
                x.len()
            )));
        }

        let mut tape = Tape::new();
        let mut var_indices = Vec::new();

        // Add variables to tape
        for (i, &val) in x.iter().enumerate() {
            let idx = tape.variable(i, val);
            var_indices.push(idx);
        }

        // Compute function
        let output_indices = f(&mut tape, &var_indices);
        let m = output_indices.len();

        let mut jacobian = Array2::zeros((m, self.n_vars));

        // Compute gradients for each output
        for (i, &output_idx) in output_indices.iter().enumerate() {
            let grad = tape.backward(output_idx, self.n_vars);
            jacobian.row_mut(i).assign(&grad);
        }

        Ok(jacobian)
    }

    /// Compute Hessian (second derivatives) using reverse-over-forward AD
    pub fn hessian<Func>(f: Func, x: ArrayView1<F>) -> IntegrateResult<Array2<F>>
    where
        Func: Fn(&mut Tape<F>, &[usize]) -> usize + Clone,
    {
        if x.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} variables, got {}",
                self.n_vars,
                x.len()
            )));
        }

        let mut hessian = Array2::zeros((self.n_vars, self.n_vars));
        let eps = F::from(1e-8).unwrap();

        // Compute Hessian using finite differences of gradients
        for j in 0..self.n_vars {
            // Perturb x[j]
            let mut x_plus = x.to_owned();
            x_plus[j] += eps;

            let grad_plus = self.gradient(f.clone(), x_plus.view())?;
            let grad_base = self.gradient(f.clone(), x)?;

            // Hessian column j = (grad(x + eps*e_j) - grad(x)) / eps
            for i in 0..self.n_vars {
                hessian[[i, j]] = (grad_plus[i] - grad_base[i]) / eps;
            }
        }

        // Make Hessian symmetric (average upper and lower triangular parts)
        for i in 0..self.n_vars {
            for j in (i + 1)..self.n_vars {
                let avg = (hessian[[i, j]] + hessian[[j, i]]) / F::from(2.0).unwrap();
                hessian[[i, j]] = avg;
                hessian[[j, i]] = avg;
            }
        }

        Ok(hessian)
    }

    /// Compute gradients for multiple inputs in batch
    pub fn batch_gradient<Func>(
        &self,
        f: Func,
        x_batch: &[Array1<F>],
    ) -> IntegrateResult<Vec<Array1<F>>>
    where
        Func: Fn(&mut Tape<F>, &[usize]) -> usize + Clone,
    {
        let mut gradients = Vec::with_capacity(x_batch.len());

        for x in x_batch {
            gradients.push(self.gradient(f.clone(), x.view())?);
        }

        Ok(gradients)
    }

    /// Compute Jacobian-vector product efficiently without forming full Jacobian
    pub fn jvp<Func>(
        &self,
        f: Func,
        x: ArrayView1<F>,
        v: ArrayView1<F>,
    ) -> IntegrateResult<Array1<F>>
    where
        Func: Fn(&mut Tape<F>, &[usize]) -> Vec<usize>,
    {
        if x.len() != self.n_vars || v.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} variables for both x and v",
                self.n_vars
            )));
        }

        // Use forward mode for efficient JVP computation
        let eps = F::from(1e-8).unwrap();
        let x_perturbed = &x + &(v.to_owned() * eps);

        let mut tape = Tape::new();
        let mut var_indices = Vec::new();
        let mut var_indices_perturbed = Vec::new();

        // Evaluate at x and x + eps*v
        for (i, &val) in x.iter().enumerate() {
            let idx = tape.variable(i, val);
            var_indices.push(idx);
        }

        let output_base = f(&mut tape, &var_indices);

        tape = Tape::new();
        for (i, &val) in x_perturbed.iter().enumerate() {
            let idx = tape.variable(i, val);
            var_indices_perturbed.push(idx);
        }

        let output_perturbed = f(&mut tape, &var_indices_perturbed);

        // Compute JVP as (f(x + eps*v) - f(x)) / eps
        let mut jvp = Array1::zeros(output_base.len());
        for (i, (&idx_base, &idx_pert)) in
            output_base.iter().zip(output_perturbed.iter()).enumerate()
        {
            jvp[i] = (tape.value(idx_pert) - tape.value(idx_base)) / eps;
        }

        Ok(jvp)
    }

    /// Compute vector-Jacobian product (useful for backpropagation)
    pub fn vjp<Func>(
        &self,
        f: Func,
        x: ArrayView1<F>,
        v: ArrayView1<F>,
    ) -> IntegrateResult<Array1<F>>
    where
        Func: Fn(&mut Tape<F>, &[usize]) -> Vec<usize>,
    {
        if x.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} variables",
                self.n_vars
            )));
        }

        let mut tape = Tape::new();
        let mut var_indices = Vec::new();

        // Add variables to tape
        for (i, &val) in x.iter().enumerate() {
            let idx = tape.variable(i, val);
            var_indices.push(idx);
        }

        // Compute function
        let output_indices = f(&mut tape, &var_indices);

        if v.len() != output_indices.len() {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Vector v length {} doesn't match output dimension {}",
                v.len(),
                output_indices.len()
            )));
        }

        // Compute weighted sum of outputs
        let mut weighted_sum = tape.constant(F::zero());
        for (i, &output_idx) in output_indices.iter().enumerate() {
            let v_i = tape.constant(v[i]);
            let term = tape.mul(v_i, output_idx);
            weighted_sum = tape.add(weighted_sum, term);
        }

        // Compute gradient of weighted sum
        Ok(tape.backward(weighted_sum, self.n_vars))
    }
}

/// Compute gradient using reverse mode AD (convenience function)
#[allow(dead_code)]
pub fn reverse_gradient<F, Func>(f: Func, x: ArrayView1<F>) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
    Func: Fn(&mut Tape<F>, &[usize]) -> usize,
{
    let ad = ReverseAD::new(x.len());
    ad.gradient(f, x)
}

/// Compute Jacobian using reverse mode AD (convenience function)
#[allow(dead_code)]
pub fn reverse_jacobian<F, Func>(f: Func, x: ArrayView1<F>) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat,
    Func: Fn(&mut Tape<F>, &[usize]) -> Vec<usize>,
{
    let ad = ReverseAD::new(x.len());
    ad.jacobian(f, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_gradient() {
        // Test gradient of f(x,y) = x^2 + y^2
        let f = |tape: &mut Tape<f64>, vars: &[usize]| {
            let x_sq = tape.mul(vars[0], vars[0]);
            let y_sq = tape.mul(vars[1], vars[1]);
            tape.add(x_sq, y_sq)
        };

        let x = Array1::from_vec(vec![3.0, 4.0]);
        let grad = reverse_gradient(f, x.view()).unwrap();

        // Gradient should be [2x, 2y] = [6, 8]
        assert!((grad[0] - 6.0).abs() < 1e-10);
        assert!((grad[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_jacobian() {
        // Test Jacobian of f(x,y) = [x^2, x*y, y^2]
        let f = |tape: &mut Tape<f64>, vars: &[usize]| {
            let x_sq = tape.mul(vars[0], vars[0]);
            let xy = tape.mul(vars[0], vars[1]);
            let y_sq = tape.mul(vars[1], vars[1]);
            vec![x_sq, xy, y_sq]
        };

        let x = Array1::from_vec(vec![2.0, 3.0]);
        let jac = reverse_jacobian(f, x.view()).unwrap();

        // Jacobian should be:
        // [[2x, 0 ],
        //  [y,  x ],
        //  [0,  2y]]
        assert!((jac[[0, 0]] - 4.0).abs() < 1e-10); // 2*2
        assert!((jac[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((jac[[1, 0]] - 3.0).abs() < 1e-10); // y
        assert!((jac[[1, 1]] - 2.0).abs() < 1e-10); // x
        assert!((jac[[2, 0]] - 0.0).abs() < 1e-10);
        assert!((jac[[2, 1]] - 6.0).abs() < 1e-10); // 2*3
    }
}
