//! Symbolic expression representation and manipulation
//!
//! This module provides the foundation for symbolic computation,
//! including expression trees, variables, and simplification rules.

use std::collections::HashMap;
use std::fmt;
use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};

/// Represents a symbolic variable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable {
    pub name: String,
    pub index: Option<usize>, // For indexed variables like y[0], y[1]
}

impl Variable {
    /// Create a new variable
    pub fn new(name: impl Into<String>) -> Self {
        Variable {
            name: name.into(),
            index: None,
        }
    }

    /// Create an indexed variable
    pub fn indexed(name: impl Into<String>, index: usize) -> Self {
        Variable {
            name: name.into(),
            index: Some(index),
        }
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.index {
            Some(idx) => write!(f, "{}[{}]", self.name, idx),
            None => write!(f, "{}", self.name),
        }
    }
}

/// Symbolic expression types
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicExpression<F: IntegrateFloat> {
    /// Constant value
    Constant(F),
    /// Variable
    Var(Variable),
    /// Addition
    Add(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// Subtraction
    Sub(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// Multiplication
    Mul(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// Division
    Div(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// Power
    Pow(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// Negation
    Neg(Box<SymbolicExpression<F>>),
    /// Sine
    Sin(Box<SymbolicExpression<F>>),
    /// Cosine
    Cos(Box<SymbolicExpression<F>>),
    /// Exponential
    Exp(Box<SymbolicExpression<F>>),
    /// Natural logarithm
    Ln(Box<SymbolicExpression<F>>),
    /// Square root
    Sqrt(Box<SymbolicExpression<F>>),
}

impl<F: IntegrateFloat> SymbolicExpression<F> {
    /// Create a constant expression
    pub fn constant(value: F) -> Self {
        SymbolicExpression::Constant(value)
    }

    /// Create a variable expression
    pub fn var(name: impl Into<String>) -> Self {
        SymbolicExpression::Var(Variable::new(name))
    }

    /// Create an indexed variable expression
    pub fn indexed_var(name: impl Into<String>, index: usize) -> Self {
        SymbolicExpression::Var(Variable::indexed(name, index))
    }

    /// Differentiate with respect to a variable
    pub fn differentiate(&self, var: &Variable) -> SymbolicExpression<F> {
        use SymbolicExpression::*;
        
        match self {
            Constant(_) => Constant(F::zero()),
            Var(v) => {
                if v == var {
                    Constant(F::one())
                } else {
                    Constant(F::zero())
                }
            }
            Add(a, b) => Add(
                Box::new(a.differentiate(var)),
                Box::new(b.differentiate(var))
            ),
            Sub(a, b) => Sub(
                Box::new(a.differentiate(var)),
                Box::new(b.differentiate(var))
            ),
            Mul(a, b) => {
                // Product rule: (a*b)' = a'*b + a*b'
                Add(
                    Box::new(Mul(
                        Box::new(a.differentiate(var)),
                        b.clone()
                    )),
                    Box::new(Mul(
                        a.clone(),
                        Box::new(b.differentiate(var))
                    ))
                )
            }
            Div(a, b) => {
                // Quotient rule: (a/b)' = (a'*b - a*b')/b²
                Div(
                    Box::new(Sub(
                        Box::new(Mul(
                            Box::new(a.differentiate(var)),
                            b.clone()
                        )),
                        Box::new(Mul(
                            a.clone(),
                            Box::new(b.differentiate(var))
                        ))
                    )),
                    Box::new(Mul(b.clone(), b.clone()))
                )
            }
            Pow(a, b) => {
                // For now, handle only constant powers
                if let Constant(n) = &**b {
                    // Power rule: (a^n)' = n * a^(n-1) * a'
                    Mul(
                        Box::new(Mul(
                            Box::new(Constant(*n)),
                            Box::new(Pow(
                                a.clone(),
                                Box::new(Constant(*n - F::one()))
                            ))
                        )),
                        Box::new(a.differentiate(var))
                    )
                } else {
                    // General case: a^b = exp(b*ln(a))
                    let exp_expr = Exp(Box::new(Mul(b.clone(), Box::new(Ln(a.clone())))));
                    exp_expr.differentiate(var)
                }
            }
            Neg(a) => Neg(Box::new(a.differentiate(var))),
            Sin(a) => {
                // (sin(a))' = cos(a) * a'
                Mul(
                    Box::new(Cos(a.clone())),
                    Box::new(a.differentiate(var))
                )
            }
            Cos(a) => {
                // (cos(a))' = -sin(a) * a'
                Neg(Box::new(Mul(
                    Box::new(Sin(a.clone())),
                    Box::new(a.differentiate(var))
                )))
            }
            Exp(a) => {
                // (e^a)' = e^a * a'
                Mul(
                    Box::new(Exp(a.clone())),
                    Box::new(a.differentiate(var))
                )
            }
            Ln(a) => {
                // (ln(a))' = a'/a
                Div(
                    Box::new(a.differentiate(var)),
                    a.clone()
                )
            }
            Sqrt(a) => {
                // (sqrt(a))' = a'/(2*sqrt(a))
                Div(
                    Box::new(a.differentiate(var)),
                    Box::new(Mul(
                        Box::new(Constant(F::from(2.0).unwrap())),
                        Box::new(Sqrt(a.clone()))
                    ))
                )
            }
        }
    }

    /// Evaluate the expression with given variable values
    pub fn evaluate(&self, values: &HashMap<Variable, F>) -> IntegrateResult<F> {
        use SymbolicExpression::*;
        
        match self {
            Constant(c) => Ok(*c),
            Var(v) => values.get(v)
                .copied()
                .ok_or_else(|| IntegrateError::ComputationError(
                    format!("Variable {} not found in values", v)
                )),
            Add(a, b) => Ok(a.evaluate(values)? + b.evaluate(values)?),
            Sub(a, b) => Ok(a.evaluate(values)? - b.evaluate(values)?),
            Mul(a, b) => Ok(a.evaluate(values)? * b.evaluate(values)?),
            Div(a, b) => {
                let b_val = b.evaluate(values)?;
                if b_val.abs() < F::epsilon() {
                    Err(IntegrateError::ComputationError("Division by zero".to_string()))
                } else {
                    Ok(a.evaluate(values)? / b_val)
                }
            }
            Pow(a, b) => Ok(a.evaluate(values)?.powf(b.evaluate(values)?)),
            Neg(a) => Ok(-a.evaluate(values)?),
            Sin(a) => Ok(a.evaluate(values)?.sin()),
            Cos(a) => Ok(a.evaluate(values)?.cos()),
            Exp(a) => Ok(a.evaluate(values)?.exp()),
            Ln(a) => {
                let a_val = a.evaluate(values)?;
                if a_val <= F::zero() {
                    Err(IntegrateError::ComputationError("Logarithm of non-positive value".to_string()))
                } else {
                    Ok(a_val.ln())
                }
            }
            Sqrt(a) => {
                let a_val = a.evaluate(values)?;
                if a_val < F::zero() {
                    Err(IntegrateError::ComputationError("Square root of negative value".to_string()))
                } else {
                    Ok(a_val.sqrt())
                }
            }
        }
    }

    /// Get all variables in the expression
    pub fn variables(&self) -> Vec<Variable> {
        use SymbolicExpression::*;
        let mut vars = Vec::new();
        
        match self {
            Constant(_) => {},
            Var(v) => vars.push(v.clone()),
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Pow(a, b) => {
                vars.extend(a.variables());
                vars.extend(b.variables());
            }
            Neg(a) | Sin(a) | Cos(a) | Exp(a) | Ln(a) | Sqrt(a) => {
                vars.extend(a.variables());
            }
        }
        
        // Remove duplicates
        vars.sort_by(|a, b| {
            match (&a.name, &b.name) {
                (n1, n2) if n1 != n2 => n1.cmp(n2),
                _ => a.index.cmp(&b.index),
            }
        });
        vars.dedup();
        vars
    }
}

/// Simplify a symbolic expression
pub fn simplify<F: IntegrateFloat>(expr: &SymbolicExpression<F>) -> SymbolicExpression<F> {
    use SymbolicExpression::*;
    
    match expr {
        // Identity simplifications
        Add(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            match (&a_simp, &b_simp) {
                (Constant(x), Constant(y)) => Constant(*x + *y),
                (Constant(x), _) if x.abs() < F::epsilon() => b_simp,
                (_, Constant(y)) if y.abs() < F::epsilon() => a_simp,
                _ => Add(Box::new(a_simp), Box::new(b_simp)),
            }
        }
        Sub(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            match (&a_simp, &b_simp) {
                (Constant(x), Constant(y)) => Constant(*x - *y),
                (_, Constant(y)) if y.abs() < F::epsilon() => a_simp,
                _ => Sub(Box::new(a_simp), Box::new(b_simp)),
            }
        }
        Mul(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            match (&a_simp, &b_simp) {
                (Constant(x), Constant(y)) => Constant(*x * *y),
                (Constant(x), _) if x.abs() < F::epsilon() => Constant(F::zero()),
                (_, Constant(y)) if y.abs() < F::epsilon() => Constant(F::zero()),
                (Constant(x), _) if (*x - F::one()).abs() < F::epsilon() => b_simp,
                (_, Constant(y)) if (*y - F::one()).abs() < F::epsilon() => a_simp,
                _ => Mul(Box::new(a_simp), Box::new(b_simp)),
            }
        }
        Div(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            match (&a_simp, &b_simp) {
                (Constant(x), Constant(y)) if y.abs() > F::epsilon() => Constant(*x / *y),
                (Constant(x), _) if x.abs() < F::epsilon() => Constant(F::zero()),
                (_, Constant(y)) if (*y - F::one()).abs() < F::epsilon() => a_simp,
                _ => Div(Box::new(a_simp), Box::new(b_simp)),
            }
        }
        Neg(a) => {
            let a_simp = simplify(a);
            match &a_simp {
                Constant(x) => Constant(-*x),
                Neg(inner) => (**inner).clone(),
                _ => Neg(Box::new(a_simp)),
            }
        }
        _ => expr.clone(),
    }
}

impl<F: IntegrateFloat> fmt::Display for SymbolicExpression<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SymbolicExpression::*;
        
        match self {
            Constant(c) => write!(f, "{}", c),
            Var(v) => write!(f, "{}", v),
            Add(a, b) => write!(f, "({} + {})", a, b),
            Sub(a, b) => write!(f, "({} - {})", a, b),
            Mul(a, b) => write!(f, "({} * {})", a, b),
            Div(a, b) => write!(f, "({} / {})", a, b),
            Pow(a, b) => write!(f, "({} ^ {})", a, b),
            Neg(a) => write!(f, "(-{})", a),
            Sin(a) => write!(f, "sin({})", a),
            Cos(a) => write!(f, "cos({})", a),
            Exp(a) => write!(f, "exp({})", a),
            Ln(a) => write!(f, "ln({})", a),
            Sqrt(a) => write!(f, "sqrt({})", a),
        }
    }
}