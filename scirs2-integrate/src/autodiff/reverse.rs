//! Reverse mode automatic differentiation (backpropagation)
//!
//! Reverse mode AD is efficient for computing gradients when the number of
//! outputs is small compared to the number of inputs.

use ndarray::{Array1, Array2, ArrayView1};
use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

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
    /// Sin
    Sin(usize),
    /// Cos
    Cos(usize),
    /// Exp
    Exp(usize),
    /// Ln
    Ln(usize),
    /// Sqrt
    Sqrt(usize),
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
    pub fn new(value: F, operation: Operation<F>) -> Self {
        TapeNode {
            value,
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
    pub fn variable(&mut self, idx: usize, value: F) -> usize {
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Variable(idx))));
        self.var_map.insert(idx, node_idx);
        node_idx
    }

    /// Add a constant to the tape
    pub fn constant(&mut self, value: F) -> usize {
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Constant(value))));
        node_idx
    }

    /// Record addition
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let value = self.nodes[a].value + self.nodes[b].value;
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Add(a, b))));
        node_idx
    }

    /// Record subtraction
    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let value = self.nodes[a].value - self.nodes[b].value;
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Sub(a, b))));
        node_idx
    }

    /// Record multiplication
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let value = self.nodes[a].value * self.nodes[b].value;
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Mul(a, b))));
        node_idx
    }

    /// Record division
    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let value = self.nodes[a].value / self.nodes[b].value;
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Div(a, b))));
        node_idx
    }

    /// Record negation
    pub fn neg(&mut self, a: usize) -> usize {
        let value = -self.nodes[a].value;
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Neg(a))));
        node_idx
    }

    /// Record power
    pub fn pow(&mut self, a: usize, n: F) -> usize {
        let value = self.nodes[a].value.powf(n);
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Pow(a, n))));
        node_idx
    }

    /// Record sin
    pub fn sin(&mut self, a: usize) -> usize {
        let value = self.nodes[a].value.sin();
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Sin(a))));
        node_idx
    }

    /// Record cos
    pub fn cos(&mut self, a: usize) -> usize {
        let value = self.nodes[a].value.cos();
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Cos(a))));
        node_idx
    }

    /// Record exp
    pub fn exp(&mut self, a: usize) -> usize {
        let value = self.nodes[a].value.exp();
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Exp(a))));
        node_idx
    }

    /// Record ln
    pub fn ln(&mut self, a: usize) -> usize {
        let value = self.nodes[a].value.ln();
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Ln(a))));
        node_idx
    }

    /// Record sqrt
    pub fn sqrt(&mut self, a: usize) -> usize {
        let value = self.nodes[a].value.sqrt();
        let node_idx = self.nodes.len();
        self.nodes.push(Rc::new(TapeNode::new(value, Operation::Sqrt(a))));
        node_idx
    }

    /// Get the value at a node
    pub fn value(&self, idx: usize) -> F {
        self.nodes[idx].value
    }

    /// Backward pass to compute gradients
    pub fn backward(&self, output_idx: usize, n_vars: usize) -> Array1<F> {
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
                Operation::Variable(_) | Operation::Constant(_) => {},
                Operation::Add(a, b) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad;
                    *self.nodes[*b].gradient.borrow_mut() += grad;
                },
                Operation::Sub(a, b) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad;
                    *self.nodes[*b].gradient.borrow_mut() -= grad;
                },
                Operation::Mul(a, b) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad * self.nodes[*b].value;
                    *self.nodes[*b].gradient.borrow_mut() += grad * self.nodes[*a].value;
                },
                Operation::Div(a, b) => {
                    let b_val = self.nodes[*b].value;
                    *self.nodes[*a].gradient.borrow_mut() += grad / b_val;
                    *self.nodes[*b].gradient.borrow_mut() -= grad * self.nodes[*a].value / (b_val * b_val);
                },
                Operation::Neg(a) => {
                    *self.nodes[*a].gradient.borrow_mut() -= grad;
                },
                Operation::Pow(a, n) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad * *n * self.nodes[*a].value.powf(*n - F::one());
                },
                Operation::Sin(a) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad * self.nodes[*a].value.cos();
                },
                Operation::Cos(a) => {
                    *self.nodes[*a].gradient.borrow_mut() -= grad * self.nodes[*a].value.sin();
                },
                Operation::Exp(a) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad * node.value;
                },
                Operation::Ln(a) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad / self.nodes[*a].value;
                },
                Operation::Sqrt(a) => {
                    *self.nodes[*a].gradient.borrow_mut() += grad / (F::from(2.0).unwrap() * node.value);
                },
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

/// Reverse mode automatic differentiation engine
pub struct ReverseAD<F: IntegrateFloat> {
    /// Number of independent variables
    n_vars: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: IntegrateFloat> ReverseAD<F> {
    /// Create a new reverse AD engine
    pub fn new(n_vars: usize) -> Self {
        ReverseAD {
            n_vars,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute gradient using reverse mode AD
    pub fn gradient<Func>(&self, f: Func, x: ArrayView1<F>) -> IntegrateResult<Array1<F>>
    where
        Func: Fn(&mut Tape<F>, &[usize]) -> usize,
    {
        if x.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(
                format!("Expected {} variables, got {}", self.n_vars, x.len())
            ));
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
    pub fn jacobian<Func>(&self, f: Func, x: ArrayView1<F>) -> IntegrateResult<Array2<F>>
    where
        Func: Fn(&mut Tape<F>, &[usize]) -> Vec<usize>,
    {
        if x.len() != self.n_vars {
            return Err(IntegrateError::DimensionMismatch(
                format!("Expected {} variables, got {}", self.n_vars, x.len())
            ));
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
}

/// Compute gradient using reverse mode AD (convenience function)
pub fn reverse_gradient<F, Func>(
    f: Func,
    x: ArrayView1<F>,
) -> IntegrateResult<Array1<F>>
where
    F: IntegrateFloat,
    Func: Fn(&mut Tape<F>, &[usize]) -> usize,
{
    let ad = ReverseAD::new(x.len());
    ad.gradient(f, x)
}

/// Compute Jacobian using reverse mode AD (convenience function)
pub fn reverse_jacobian<F, Func>(
    f: Func,
    x: ArrayView1<F>,
) -> IntegrateResult<Array2<F>>
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